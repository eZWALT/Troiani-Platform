from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from troiani_platform.config import PlatformConfig, TimeWindow
from troiani_platform.infra.notation import gpu_ref
from troiani_platform.infra.storage import storage_report
from troiani_platform.jobs.skypilot import spec_from_task_yaml
from troiani_platform.policy.windows import apply_aggressiveness, default_presets, in_window, merge_presets
from troiani_platform.errors import CheckpointError, JobSpecError, NotFound, PolicyError, TransitionError
from troiani_platform.log import get_logger
from troiani_platform.models import (
    ACTIVE_STATES,
    elapsed_seconds,
    Artifact,
    CheckpointKind,
    CheckpointRecord,
    CheckpointState,
    Event,
    GPUResource,
    IntentSignal,
    Job,
    JobSpec,
    JobState,
    Occupancy,
    Run,
    WorkerRecord,
    new_id,
    parse_ts,
    utcnow,
)
from troiani_platform.monitoring.annoyance import compute_annoyance
from troiani_platform.monitoring.metrics import InfraMetrics
from troiani_platform.scheduler.scheduler import CooperativeScheduler
from troiani_platform.state.store import StateStore
from troiani_platform.state.transitions import apply_transition
from troiani_platform.tracking.environment import capture_environment, git_sha
from troiani_platform.tracking.mlflow_sink import MLflowSink
from troiani_platform.tracking.reproduce import reproduce_run
from troiani_platform.tracking.tensorboard_sink import TensorBoardSink
from troiani_platform.training.checkpoint import CheckpointManager, manifest_to_report
from troiani_platform.training.dataset import fingerprint_dataset
from troiani_platform.training.recovery import latest_valid, select_healthy

log = get_logger("troiani.platform.control")


class PlatformService:
    def __init__(self, config: PlatformConfig, store: StateStore | None = None) -> None:
        self.config = config
        self.config.ensure_dirs()
        self.store = store or StateStore(config.paths.store)
        self.scheduler = CooperativeScheduler(config.policy)
        self.checkpoints = CheckpointManager(config.paths.checkpoints, config.policy.retention)
        self.metrics = InfraMetrics.from_dict(self.store.get_kv("infra_metrics"))
        self.mlflow = MLflowSink(config.mlflow_tracking_uri)
        self.tensorboard = TensorBoardSink(config.paths.tensorboard)
        self._pending_actions: dict[str, list[dict[str, Any]]] = {}
        self._last_tick = datetime.now(timezone.utc)
        saved = self.store.get_kv("policy_override")
        if isinstance(saved, dict):
            try:
                self.update_policy(saved)
            except Exception:
                pass
        if not self.config.policy.windows:
            self.config.policy.windows = default_presets()
        else:
            self.config.policy.windows = merge_presets(self.config.policy.windows)

    def emit(self, type_: str, **kwargs: Any) -> Event:
        event = Event(type=type_, **kwargs)
        self.store.append_event(event)
        return event

    def submit_from_yaml(self, raw: dict[str, Any] | str) -> Job:
        if isinstance(raw, str):
            parsed = yaml.safe_load(raw) or {}
        else:
            parsed = raw
        if not isinstance(parsed, dict):
            raise JobSpecError("YAML root must be a mapping")
        inner = parsed.get("yaml")
        if isinstance(inner, str) and "command" not in parsed and "run" not in parsed and "entrypoint" not in parsed:
            parsed = yaml.safe_load(inner) or {}
        return self.submit(spec_from_task_yaml(parsed if isinstance(parsed, dict) else {}))

    def submit(self, spec: JobSpec | dict[str, Any]) -> Job:
        if isinstance(spec, dict) and (
            isinstance(spec.get("command"), str) or spec.get("run") or spec.get("entrypoint") or spec.get("resources")
        ):
            spec = spec_from_task_yaml(spec)
        job_spec = spec if isinstance(spec, JobSpec) else JobSpec.from_dict(spec)
        job = Job(id=new_id("job"), spec=job_spec, state=JobState.PENDING)
        dataset = fingerprint_dataset(None, job_spec.dataset)
        run = Run(
            id=new_id("run"),
            job_id=job.id,
            experiment=job_spec.experiment or job_spec.name,
            git_sha=git_sha(),
            config=job_spec.to_dict(),
            environment=capture_environment(),
            dataset=dataset.to_dict() if dataset else None,
        )
        job.run_id = run.id
        self.store.put_job(job)
        self.store.put_run(run)
        self.emit("JOB_SUBMITTED", job_id=job.id, run_id=run.id, payload={"name": job_spec.name})
        try:
            self.mlflow.start_run(
                run.id,
                run.experiment,
                {k: v for k, v in job_spec.to_dict().items() if k != "command"},
                {"job_id": job.id, "git": run.git_sha or ""},
            )
        except Exception:
            pass
        self.tick()
        return self.store.get_job(job.id)

    def get_job(self, job_id: str) -> Job:
        return self.store.get_job(job_id)

    def _set_state(self, job: Job, target: JobState, message: str = "") -> Job:
        now = utcnow()
        if job.state in ACTIVE_STATES and target not in ACTIVE_STATES and job.active_since:
            job = job.touch(trained_s=job.trained_s + elapsed_seconds(job.active_since, now=now), active_since=None)
        job = apply_transition(job, target, message)
        if target in ACTIVE_STATES and job.active_since is None:
            job = job.touch(active_since=now)
        return self.store.put_job(job)

    def cancel(self, job_id: str) -> Job:
        job = self.store.get_job(job_id)
        if job.state in ACTIVE_STATES:
            self._queue_action(job.worker_id, {"kind": "STOP", "job_id": job.id})
        job = self._set_state(job, JobState.CANCELLED, "cancelled by user")
        self._release_gpus(job)
        self.emit("JOB_CANCELLED", job_id=job.id, run_id=job.run_id)
        return job

    def pause(self, job_id: str) -> Job:
        job = self.store.get_job(job_id)
        if job.state == JobState.PAUSED:
            return job
        if job.state in {JobState.PENDING, JobState.QUEUED, JobState.PREEMPTED}:
            job = self._set_state(job, JobState.PAUSED, "paused")
        elif job.state in {JobState.SCHEDULED, JobState.STARTING, JobState.RESUMING}:
            self._queue_action(job.worker_id, {"kind": "STOP", "job_id": job.id})
            job = self._set_state(job, JobState.QUEUED, "pause before run")
            job = self._set_state(job, JobState.PAUSED, "paused")
            self._release_gpus(job)
            job = job.touch(assigned_gpus=[], assigned_node=None, worker_id=None)
            self.store.put_job(job)
        elif job.state == JobState.RUNNING:
            job = job.touch(preempt_requested=True)
            self.store.put_job(job)
            self._queue_action(job.worker_id, {"kind": "PREEMPT", "job_id": job.id})
            job = self._set_state(job, JobState.CHECKPOINTING, "pause requested")
            job = self._set_state(job, JobState.PAUSED, "paused")
        elif job.state == JobState.CHECKPOINTING:
            job = self._set_state(job, JobState.PAUSED, "paused")
        else:
            job = self._set_state(job, JobState.PAUSED, "paused")
        self.emit("JOB_PAUSED", job_id=job.id, run_id=job.run_id)
        return self.store.get_job(job.id)

    def resume(self, job_id: str) -> Job:
        job = self.store.get_job(job_id)
        if job.state in ACTIVE_STATES:
            return job
        if job.state in {JobState.COMPLETED, JobState.CANCELLED}:
            return job
        job = job.touch(message="resume:user", preempt_requested=False)
        if job.state == JobState.PAUSED:
            job = self._set_state(job, JobState.QUEUED, "resume:user")
        elif job.state in {JobState.PREEMPTED, JobState.FAILED}:
            job = self._set_state(job, JobState.QUEUED, "resume:user")
        self.emit("JOB_RESUMED", job_id=job.id, run_id=job.run_id)
        self.tick()
        return self.store.get_job(job.id)

    def preempt(self, job_id: str, reason: str = "manual") -> Job:
        job = self.store.get_job(job_id)
        if job.state not in ACTIVE_STATES:
            return job
        job = job.touch(preempt_requested=True, message=reason)
        self.store.put_job(job)
        self._queue_action(job.worker_id, {"kind": "PREEMPT", "job_id": job.id})
        self.emit("PREEMPT_REQUESTED", job_id=job.id, run_id=job.run_id, payload={"reason": reason})
        if job.state == JobState.RUNNING:
            job = self._set_state(job, JobState.CHECKPOINTING, reason)
            self.store.put_job(job)
        return self.store.get_job(job.id)

    def release_all(self) -> list[Job]:
        released = []
        for job in self.store.list_jobs():
            if job.state in ACTIVE_STATES and job.spec.preemptible:
                released.append(self.preempt(job.id, reason="release-all"))
        self.emit("RELEASE_ALL", payload={"count": len(released)})
        return released

    def _queue_action(self, worker_id: str | None, action: dict[str, Any]) -> None:
        if not worker_id:
            return
        self._pending_actions.setdefault(worker_id, []).append(action)

    def _release_gpus(self, job: Job) -> None:
        for gpu in self.store.list_gpus():
            if gpu.job_id == job.id:
                self.store.put_gpu(
                    GPUResource.from_dict(
                        {
                            **gpu.to_dict(),
                            "job_id": None,
                            "occupancy": Occupancy.AVAILABLE.value
                            if not gpu.researcher_present
                            else Occupancy.RESEARCHER.value,
                        }
                    )
                )
        if job.assigned_gpus:
            self.emit("GPU_RELEASED", job_id=job.id, payload={"gpus": list(job.assigned_gpus)})

    def _heartbeat_age_s(self, last: str | None, now: datetime | None = None) -> float:
        current = now or datetime.now(timezone.utc)
        parsed = parse_ts(last)
        if parsed is None:
            return float("inf")
        return max(0.0, (current - parsed).total_seconds())

    def _stale_workers(self, now: datetime | None = None) -> list[tuple[WorkerRecord, float]]:
        current = now or datetime.now(timezone.utc)
        threshold = self.config.policy.worker_stale_s
        stale: list[tuple[WorkerRecord, float]] = []
        for worker in self.store.list_workers():
            age = self._heartbeat_age_s(worker.last_heartbeat, current)
            if age > threshold:
                stale.append((worker, age))
        return stale

    def _job_stale_worker(
        self,
        job: Job,
        stale_by_id: dict[str, WorkerRecord],
        stale_by_node: dict[str, WorkerRecord],
    ) -> WorkerRecord | None:
        if job.worker_id and job.worker_id in stale_by_id:
            return stale_by_id[job.worker_id]
        pending = not job.worker_id or str(job.worker_id).endswith("-pending")
        if pending and job.assigned_node and job.assigned_node in stale_by_node:
            return stale_by_node[job.assigned_node]
        return None

    def _preempt_to_queue(self, job: Job, message: str) -> Job:
        """Walk only valid transitions to PREEMPTED then QUEUED. No remote kills."""
        try:
            if job.state == JobState.RESUMING:
                job = self._set_state(job, JobState.CHECKPOINTING, message)
            if job.state in {
                JobState.RUNNING,
                JobState.STARTING,
                JobState.SCHEDULED,
                JobState.CHECKPOINTING,
            }:
                job = self._set_state(job, JobState.PREEMPTED, message)
            if job.state == JobState.PREEMPTED:
                job = self._set_state(job, JobState.QUEUED, message)
        except TransitionError as exc:
            log.warning(
                "stale reclaim skipped invalid transition",
                extra={"job_id": job.id, "state": job.state.value, "error": str(exc)},
            )
        return job

    def _reclaim_orphaned_job(self, job: Job, reason: str) -> Job:
        # Worker is unreachable. Do not queue STOP/KILL/PREEMPT and do not
        # invent remote researcher PID kills. Local watchdog still releases
        # Troiani children if the worker process is alive but partitioned.
        job = self._preempt_to_queue(job, reason)
        self._release_gpus(job)
        job = job.touch(
            assigned_gpus=[],
            assigned_node=None,
            worker_id=None,
            preempt_requested=False,
            pid=None,
            message=reason,
        )
        return self.store.put_job(job)

    def _reclaim_stale_workers(self) -> list[WorkerRecord]:
        stale = self._stale_workers()
        if not stale:
            return []
        stale_by_id = {worker.id: worker for worker, _ in stale}
        stale_by_node = {worker.node: worker for worker, _ in stale}
        reclaimed: dict[str, list[str]] = {worker.id: [] for worker, _ in stale}
        seen_jobs: set[str] = set()

        for job in self.store.list_jobs():
            if job.state not in ACTIVE_STATES:
                continue
            owner = self._job_stale_worker(job, stale_by_id, stale_by_node)
            if owner is None:
                continue
            self._reclaim_orphaned_job(job, "worker stale")
            reclaimed.setdefault(owner.id, []).append(job.id)
            seen_jobs.add(job.id)

        for gpu in self.store.list_gpus():
            if not gpu.job_id or gpu.node not in stale_by_node:
                continue
            if gpu.job_id in seen_jobs:
                continue
            try:
                job = self.store.get_job(gpu.job_id)
            except NotFound:
                self.store.put_gpu(
                    GPUResource.from_dict(
                        {
                            **gpu.to_dict(),
                            "job_id": None,
                            "occupancy": Occupancy.AVAILABLE.value
                            if not gpu.researcher_present
                            else Occupancy.RESEARCHER.value,
                        }
                    )
                )
                continue
            if job.state not in ACTIVE_STATES:
                self._release_gpus(job)
                continue
            owner = stale_by_node[gpu.node]
            self._reclaim_orphaned_job(job, "worker stale")
            reclaimed.setdefault(owner.id, []).append(job.id)
            seen_jobs.add(job.id)

        for worker, age in stale:
            job_ids = reclaimed.get(worker.id, [])
            became_stale = worker.status != "stale"
            if became_stale:
                self.store.put_worker(
                    WorkerRecord.from_dict(
                        {**worker.to_dict(), "status": "stale", "message": "heartbeat stale"}
                    )
                )
            if became_stale or job_ids:
                self.emit(
                    "WORKER_STALE",
                    worker_id=worker.id,
                    payload={
                        "node": worker.node,
                        "heartbeat_age_s": None if age == float("inf") else age,
                        "job_ids": job_ids,
                    },
                )
            self._pending_actions.pop(worker.id, None)
            self._pending_actions.pop(f"{worker.node}-pending", None)
        return [worker for worker, _ in stale]

    def tick(self) -> list[dict[str, Any]]:
        stale_workers = self._reclaim_stale_workers()
        stale_nodes = {worker.node for worker in stale_workers}
        jobs = self.store.list_jobs()
        gpus = self.store.list_gpus()
        now = datetime.now(timezone.utc)
        dt = (now - self._last_tick).total_seconds()
        self.metrics.observe_gpus(gpus, max(dt, 0))
        self.metrics.observe_util(gpus, utcnow())
        self.store.set_kv("infra_metrics", self.metrics.to_dict())
        self._last_tick = now
        schedulable = [gpu for gpu in gpus if gpu.node not in stale_nodes]
        cooldown = {
            uuid
            for job in jobs
            if job.cooldown_until
            for uuid in job.assigned_gpus
        }
        decisions = self.scheduler.schedule(jobs, schedulable, now=now, cooldown_gpus=cooldown)
        applied = []
        by_id = {job.id: job for job in jobs}
        for decision in decisions:
            job = by_id.get(decision.job_id)
            if not job:
                continue
            if decision.kind == "PREEMPT":
                self.preempt(job.id, reason=decision.reason)
                self.metrics.preemptions += 1
            elif decision.kind == "ALLOCATE" and decision.placement:
                self._allocate(job, decision.placement.node, decision.placement.gpu_uuids)
            elif decision.kind == "HOLD":
                self.emit("POLICY_HOLD", job_id=job.id, payload={"reason": decision.reason})
            applied.append(decision.to_dict())
        return applied

    def _allocate(self, job: Job, node: str, gpu_uuids: list[str]) -> Job:
        workers = [w for w in self.store.list_workers() if w.node == node]
        worker_id = workers[0].id if workers else f"{node}-pending"
        resuming = bool(job.last_checkpoint_id or job.last_step)
        target = JobState.RESUMING if resuming and job.state in {JobState.QUEUED, JobState.PREEMPTED, JobState.PENDING} else JobState.SCHEDULED
        if job.state in {JobState.PREEMPTED}:
            job = self._set_state(job, JobState.QUEUED, "requeue after preempt")
        if job.state == JobState.PENDING:
            job = self._set_state(job, JobState.QUEUED, "admitted")
        if job.state == JobState.QUEUED and target == JobState.RESUMING:
            job = self._set_state(job, JobState.SCHEDULED, "resume placement")
            job = self._set_state(job, JobState.RESUMING, "resuming")
        elif job.state in {JobState.QUEUED, JobState.PENDING}:
            job = self._set_state(job, JobState.SCHEDULED, "placed")
        job = job.touch(assigned_gpus=list(gpu_uuids), assigned_node=node, worker_id=worker_id, preempt_requested=False)
        self.store.put_job(job)
        for gpu in self.store.list_gpus():
            if gpu.uuid in gpu_uuids:
                self.store.put_gpu(
                    GPUResource.from_dict({**gpu.to_dict(), "job_id": job.id, "occupancy": Occupancy.TROIANI.value, "worker_id": worker_id})
                )
        run = self.store.get_run(job.run_id) if job.run_id else None
        if run:
            run.hardware = {"node": node, "gpus": gpu_uuids}
            run.status = "scheduled"
            run.started_at = run.started_at or utcnow()
            self.store.put_run(run)
        self.emit("JOB_SCHEDULED", job_id=job.id, run_id=job.run_id, payload={"node": node, "gpus": gpu_uuids})
        action = {
            "kind": "START",
            "job_id": job.id,
            "job": job.to_dict(),
            "run": run.to_dict() if run else {},
            "local_indexes": self._local_indexes(node, gpu_uuids),
        }
        self._queue_action(worker_id, action)
        return job

    def _local_indexes(self, node: str, uuids: list[str]) -> list[int]:
        lookup = {gpu.uuid: gpu.index for gpu in self.store.list_gpus() if gpu.node == node}
        return [lookup.get(u, 0) for u in uuids]

    def _reconcile_worker_jobs(self, worker: WorkerRecord, reported: dict[str, Any]) -> None:
        """If the worker still heartbeats but the trainer is gone, do not sit in CHECKPOINTING."""
        for job in self.store.list_jobs():
            if job.state not in ACTIVE_STATES:
                continue
            owns = job.worker_id == worker.id or (
                not job.worker_id and job.assigned_node == worker.node
            )
            if not owns:
                continue
            info = reported.get(job.id) or {}
            listed = job.id in reported
            alive = bool(info.get("alive"))
            if job.state in {JobState.SCHEDULED, JobState.STARTING, JobState.RESUMING} and not listed:
                continue
            if listed and alive:
                continue
            if not listed and job.state not in {JobState.RUNNING, JobState.CHECKPOINTING}:
                continue
            tail = str(info.get("output_tail") or "worker reported process gone")
            code = info.get("exit_code")
            if code is None:
                code = 75 if (job.preempt_requested or job.state == JobState.CHECKPOINTING) else 1
            try:
                self.job_exit(
                    {
                        "job_id": job.id,
                        "exit_code": int(code),
                        "success": bool(info.get("success")),
                        "output_tail": tail,
                    }
                )
            except (NotFound, TransitionError):
                log.warning("reconcile exit failed", extra={"job_id": job.id, "state": job.state.value})

    def heartbeat(self, payload: dict[str, Any]) -> dict[str, Any]:
        worker = WorkerRecord(
            id=str(payload.get("worker_id") or payload.get("node") or "worker"),
            node=str(payload.get("node") or "unknown"),
            hostname=str(payload.get("hostname") or ""),
            last_heartbeat=utcnow(),
            gpus=[g.get("uuid") for g in payload.get("gpus") or [] if g.get("uuid")],
            status="alive",
        )
        self.store.put_worker(worker)
        for raw in payload.get("gpus") or []:
            gpu = GPUResource.from_dict(raw)
            existing = next((g for g in self.store.list_gpus() if g.uuid == gpu.uuid), None)
            if existing and existing.job_id and gpu.occupancy != Occupancy.RESEARCHER:
                gpu = GPUResource.from_dict({**gpu.to_dict(), "job_id": existing.job_id, "occupancy": Occupancy.TROIANI.value})
            draining = set(self.store.get_kv("draining_gpus", []) or [])
            if gpu.uuid in draining and gpu.occupancy != Occupancy.RESEARCHER and not gpu.researcher_present:
                gpu = GPUResource.from_dict({**gpu.to_dict(), "occupancy": Occupancy.DRAINING.value})
            self.store.put_gpu(gpu)
        for job_id, raw in (payload.get("logs") or {}).items():
            if job_id:
                self.ingest_job_log(str(job_id), raw, worker.node)
        for raw in payload.get("intents") or []:
            signal = IntentSignal.from_dict(raw)
            fingerprint = f"{signal.kind.value}:{signal.node}:{signal.gpu_uuid}:{signal.username}:{signal.detail[:80]}"
            seen = set(self.store.get_kv("intent_seen", []) or [])
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            self.store.set_kv("intent_seen", sorted(seen)[-200:])
            self.store.put_intent(signal)
            self.emit("INTENT_SIGNAL", payload=signal.to_dict(), source="worker")
        self.emit("WORKER_HEARTBEAT", worker_id=worker.id, payload={"node": worker.node})
        if payload.get("infra"):
            self.store.set_kv(f"node_infra:{worker.node}", payload.get("infra"))
        self._reconcile_worker_jobs(worker, payload.get("jobs") or {})
        self.tick()
        actions = self._pending_actions.pop(worker.id, [])
        # Also deliver actions queued under node-pending once a real worker appears.
        pending_key = f"{worker.node}-pending"
        actions.extend(self._pending_actions.pop(pending_key, []))
        return {"ok": True, "actions": actions, "worker_id": worker.id}

    def record_checkpoint(self, payload: dict[str, Any]) -> CheckpointRecord:
        report = manifest_to_report(payload, path=payload.get("path") or "")
        ckpt_id = str(report.get("id") or "").strip()
        run_id = str(report.get("run_id") or "").strip()
        job_id = str(report.get("job_id") or "").strip()
        if not ckpt_id or not run_id or not job_id:
            raise CheckpointError("checkpoint report requires id, run_id, and job_id")
        kind_raw = report.get("kind") or CheckpointKind.REGULAR
        try:
            kind = kind_raw if isinstance(kind_raw, CheckpointKind) else CheckpointKind(str(kind_raw))
        except ValueError as exc:
            raise CheckpointError(f"invalid checkpoint kind: {kind_raw}") from exc
        record = CheckpointRecord(
            id=ckpt_id,
            run_id=run_id,
            job_id=job_id,
            step=int(report.get("step") or 0),
            epoch=int(payload.get("epoch") or (payload.get("checkpoint") or {}).get("epoch") or 0),
            kind=kind,
            state=CheckpointState.VALID,
            path=str(report.get("path") or ""),
            healthy=bool(report.get("healthy")),
            metric=payload.get("metric"),
            checksum=str(report.get("checksum") or ""),
            manifest=dict(payload.get("manifest") or payload),
        )
        self.store.put_checkpoint(record)
        try:
            job = self.store.get_job(job_id)
            job = self._apply_checkpoint_pointer(job, record)
            self.store.put_job(job)
        except NotFound:
            pass
        if run_id:
            try:
                run = self.store.get_run(run_id)
                if record.id not in run.checkpoints:
                    run.checkpoints = [*run.checkpoints, record.id]
                    self.store.put_run(run)
            except NotFound:
                pass
        self.emit(
            "CHECKPOINT_RECORDED",
            job_id=job_id,
            run_id=run_id,
            payload={"id": record.id, "step": record.step, "path": record.path},
        )
        return record

    def _apply_checkpoint_pointer(self, job: Job, record: CheckpointRecord) -> Job:
        if job.last_checkpoint_id:
            try:
                current = self.store.get_checkpoint(job.last_checkpoint_id)
                if record.step < current.step:
                    return job
            except NotFound:
                pass
        return job.touch(last_checkpoint_id=record.id, last_step=max(job.last_step, record.step))

    def _latest_recorded_checkpoint(self, run_id: str | None) -> CheckpointRecord | None:
        if not run_id:
            return None
        return latest_valid(self.store.list_checkpoints(run_id))

    def _ingest_local_checkpoint(self, job: Job) -> CheckpointRecord | None:
        if not job.run_id:
            return None
        latest = self.checkpoints.latest_valid(job.run_id)
        if not latest:
            return None
        path = self.config.paths.checkpoints / job.run_id / latest["checkpoint"]["id"]
        return self.record_checkpoint({**manifest_to_report(latest, path), "manifest": latest})

    def _sync_job_checkpoint(self, job: Job) -> Job:
        record = self._latest_recorded_checkpoint(job.run_id) or self._ingest_local_checkpoint(job)
        if record is None:
            return self.store.get_job(job.id)
        job = self.store.get_job(job.id)
        job = self._apply_checkpoint_pointer(job, record)
        return self.store.put_job(job)

    def job_exit(self, payload: dict[str, Any]) -> Job:
        job = self.store.get_job(str(payload["job_id"]))
        if job.state in {JobState.COMPLETED, JobState.CANCELLED}:
            return self._sync_job_checkpoint(job)
        code = int(payload.get("exit_code") if payload.get("exit_code") is not None else 1)
        tail = str(payload.get("output_tail") or job.message or "")
        if job.run_id:
            done = self.config.paths.checkpoints / job.run_id / "DONE"
            if done.exists():
                try:
                    marker = json.loads(done.read_text())
                    if marker.get("preempt"):
                        code = 75
                    elif code != 0:
                        code = 0
                except json.JSONDecodeError:
                    pass
        preempted = bool(job.preempt_requested or code == 75 or "preempted at step=" in tail)
        finished = bool(
            payload.get("success")
            or code == 0
            or "completed step=" in tail
        )
        if finished:
            preempted = False
        elif preempted:
            self._finish_preempt(job)
            return self.store.get_job(job.id)
        latest = self.checkpoints.latest_valid(job.run_id) if job.run_id else None
        recorded = self._latest_recorded_checkpoint(job.run_id)
        if latest and code not in {-9, -15}:
            kind = str(latest.get("kind") or "")
            step = int((latest.get("checkpoint") or {}).get("step") or 0)
            if kind in {"shutdown", "regular", "milestone", "preemption"} and step > 0:
                finished = True
        elif recorded and code not in {-9, -15} and recorded.step > 0:
            if recorded.kind in {
                CheckpointKind.SHUTDOWN,
                CheckpointKind.REGULAR,
                CheckpointKind.MILESTONE,
                CheckpointKind.PREEMPTION,
            }:
                finished = True
        if finished:
            if job.state == JobState.FAILED:
                job = self._set_state(job, JobState.QUEUED, "repair false failure")
            job = self._set_state(job, JobState.COMPLETED, f"exited {code}")
            self._release_gpus(job)
            job = job.touch(assigned_gpus=[], ended_at=utcnow())
            self.store.put_job(job)
            job = self._sync_job_checkpoint(job)
            self.emit("JOB_COMPLETED", job_id=job.id, run_id=job.run_id)
        else:
            job = self._set_state(job, JobState.FAILED, f"exit {code}")
            self._release_gpus(job)
            job = job.touch(assigned_gpus=[], ended_at=utcnow(), message=tail[:500])
            self.store.put_job(job)
            job = self._sync_job_checkpoint(job)
            self.emit("JOB_FAILED", job_id=job.id, run_id=job.run_id, payload={"code": code})
        return self.store.get_job(job.id)

    def _finish_preempt(self, job: Job) -> Job:
        job = self._sync_job_checkpoint(job)
        if job.state == JobState.RUNNING:
            job = self._set_state(job, JobState.CHECKPOINTING, "preempt checkpoint")
        if job.state in {JobState.STARTING, JobState.RESUMING}:
            job = self._set_state(job, JobState.CHECKPOINTING, "preempt before running")
        if job.state == JobState.SCHEDULED:
            job = self._set_state(job, JobState.PREEMPTED, "preempted before start")
        if job.state == JobState.CHECKPOINTING:
            job = self._set_state(job, JobState.PREEMPTED, "preempted")
        if job.state == JobState.PREEMPTED:
            until = (datetime.now(timezone.utc) + timedelta(seconds=self.config.policy.cooldown_s)).isoformat()
            job = self._set_state(job, JobState.QUEUED, "requeued")
            job = job.touch(
                assigned_gpus=[],
                assigned_node=None,
                worker_id=None,
                preempt_requested=False,
                cooldown_until=until,
            )
            self.store.put_job(job)
        self._release_gpus(job)
        self.emit("PREEMPTED", job_id=job.id, run_id=job.run_id)
        return job

    def record_metrics(self, payload: dict[str, Any]) -> None:
        run_id = payload.get("run_id")
        job_id = payload.get("job_id")
        step = payload.get("step")
        logged: dict[str, float] = {}
        for key in ("loss", "throughput", "lr", "grad_norm", "tokens_per_sec", "val_loss"):
            if payload.get(key) is None:
                continue
            value = float(payload[key])
            logged[key] = value
            self.store.log_metric(key, value, job_id=job_id, run_id=run_id, step=step)
            if run_id and step is not None:
                self.tensorboard.log(str(run_id), key, value, int(step))
        tokens_delta = payload.get("tokens_delta")
        tokens_total = payload.get("tokens_ingested")
        if tokens_delta is not None:
            self.metrics.observe_tokens(float(tokens_delta))
            self.store.log_metric("tokens_delta", float(tokens_delta), job_id=job_id, run_id=run_id, step=step)
        if logged and run_id:
            try:
                run = self.store.get_run(str(run_id))
                self.mlflow.log_metrics(run.experiment, str(run_id), logged, step=int(step) if step is not None else None)
            except Exception:
                pass
        if job_id and payload.get("loss") is not None:
            try:
                job = self.store.get_job(str(job_id))
                became_running = False
                if job.state == JobState.SCHEDULED:
                    job = self._set_state(job, JobState.STARTING, "process up")
                    job = self._set_state(job, JobState.RUNNING, "first metric")
                    became_running = True
                elif job.state in {JobState.STARTING, JobState.RESUMING}:
                    job = self._set_state(job, JobState.RUNNING, "first metric")
                    became_running = True
                ingested = job.tokens_ingested
                if tokens_total is not None:
                    ingested = int(tokens_total)
                elif tokens_delta is not None:
                    ingested = job.tokens_ingested + int(tokens_delta)
                tps = payload.get("tokens_per_sec")
                job = job.touch(
                    last_step=int(step or job.last_step),
                    last_loss=float(payload["loss"]),
                    tokens_ingested=ingested,
                    tokens_per_sec=float(tps) if tps is not None else job.tokens_per_sec,
                )
                self.store.put_job(job)
                if job.run_id:
                    try:
                        run = self.store.get_run(job.run_id)
                        run.tokens_ingested = ingested
                        run.tokens_per_sec = job.tokens_per_sec
                        extra = {"loss": float(payload["loss"]), "step": float(step or 0)}
                        if payload.get("val_loss") is not None:
                            extra["val_loss"] = float(payload["val_loss"])
                        run.metrics = {**run.metrics, **extra}
                        if became_running:
                            run.status = "running"
                            run.started_at = run.started_at or utcnow()
                        self.store.put_run(run)
                    except NotFound:
                        pass
                if became_running:
                    self.emit("JOB_STARTED", job_id=job.id, run_id=job.run_id)
            except NotFound:
                pass
        if payload.get("anomalies"):
            self.record_anomaly(payload)

    def record_anomaly(self, payload: dict[str, Any]) -> None:
        job_id = str(payload["job_id"])
        kinds = payload.get("kinds") or payload.get("anomalies") or []
        self.emit("ANOMALY_DETECTED", job_id=job_id, run_id=payload.get("run_id"), payload={"kinds": kinds})
        try:
            job = self.store.get_job(job_id)
        except NotFound:
            return
        if job.state == JobState.RUNNING:
            job = self._set_state(job, JobState.ANOMALY, ",".join(kinds))
            job = self._set_state(job, JobState.PAUSED, "anomaly pause")
            self._queue_action(job.worker_id, {"kind": "PREEMPT", "job_id": job.id})
            self.store.put_job(job)
        if self.config.policy.auto_rollback and job.rollback_count < self.config.policy.max_rollbacks:
            self.rollback(job.id)

    def rollback(self, job_id: str) -> Job:
        job = self.store.get_job(job_id)
        records = self.store.list_checkpoints(job.run_id) if job.run_id else []
        chosen = select_healthy(records) or latest_valid(records)
        self.emit("ROLLBACK_STARTED", job_id=job.id, run_id=job.run_id, payload={"checkpoint": chosen.id if chosen else None})
        job = job.touch(rollback_count=job.rollback_count + 1, last_checkpoint_id=chosen.id if chosen else job.last_checkpoint_id)
        if job.state == JobState.ANOMALY:
            job = self._set_state(job, JobState.PAUSED, "rollback")
        if job.state == JobState.PAUSED:
            job = self._set_state(job, JobState.QUEUED, "rollback resume")
        self.store.put_job(job)
        return job

    def policy_dict(self) -> dict[str, Any]:
        policy = self.config.policy
        return {
            "timezone": policy.timezone,
            "windows": [w.asdict() for w in policy.windows],
            "max_troiani_gpus": policy.max_troiani_gpus,
            "max_jobs": policy.max_jobs,
            "max_gpus_per_job": policy.max_gpus_per_job,
            "max_runtime_s": policy.max_runtime_s,
            "preempt_grace_s": policy.preempt_grace_s,
            "cooldown_s": policy.cooldown_s,
            "auto_rollback": policy.auto_rollback,
            "stall_seconds": policy.stall_seconds,
            "memory_busy_gb": policy.memory_busy_gb,
            "stop_all": bool(policy.stop_all),
            "aggressiveness": policy.aggressiveness or "mid",
        }

    def update_policy(self, patch: dict[str, Any]) -> dict[str, Any]:
        policy = self.config.policy
        ints = (
            "max_troiani_gpus",
            "max_jobs",
            "max_gpus_per_job",
            "max_runtime_s",
            "preempt_grace_s",
            "cooldown_s",
        )
        for key in ints:
            if key in patch and patch[key] is not None:
                value = int(patch[key])
                if value < 0:
                    raise PolicyError(f"{key} must be >= 0")
                setattr(policy, key, value)
        if "auto_rollback" in patch:
            policy.auto_rollback = bool(patch["auto_rollback"])
        if "stall_seconds" in patch and patch["stall_seconds"] is not None:
            policy.stall_seconds = float(patch["stall_seconds"])
        if "memory_busy_gb" in patch and patch["memory_busy_gb"] is not None:
            policy.memory_busy_gb = float(patch["memory_busy_gb"])
        if "windows" in patch:
            windows = []
            for raw in patch["windows"] or []:
                windows.append(TimeWindow.from_mapping(raw))
            policy.windows = windows
        if "timezone" in patch and patch["timezone"]:
            policy.timezone = str(patch["timezone"])
        if "aggressiveness" in patch and patch["aggressiveness"]:
            if "max_troiani_gpus" in patch or "memory_busy_gb" in patch:
                policy.aggressiveness = str(patch["aggressiveness"]).strip().lower() or "mid"
            else:
                apply_aggressiveness(policy, str(patch["aggressiveness"]))
        if "stop_all" in patch:
            policy.stop_all = bool(patch["stop_all"])
        self.store.set_kv("policy_override", self.policy_dict())
        self.emit("POLICY_UPDATED", payload=self.policy_dict(), source="ui")
        return self.policy_dict()

    def stop_all(self) -> dict[str, Any]:
        self.config.policy.stop_all = True
        released = self.release_all()
        self.store.set_kv("policy_override", self.policy_dict())
        self.emit("POLICY_STOP_ALL", payload={"released": len(released)})
        return {"ok": True, "stop_all": True, "released": [job.id for job in released], "policy": self.policy_dict()}

    def resume_all(self) -> dict[str, Any]:
        self.config.policy.stop_all = False
        self.store.set_kv("policy_override", self.policy_dict())
        self.emit("POLICY_RESUME", payload=self.policy_dict())
        self.tick()
        return {"ok": True, "stop_all": False, "policy": self.policy_dict()}

    def drain_gpu(self, uuid: str) -> dict[str, Any]:
        gpu = next((g for g in self.store.list_gpus() if g.uuid == uuid), None)
        if gpu is None:
            raise NotFound(f"gpu {uuid} not found")
        if gpu.researcher_present or gpu.occupancy == Occupancy.RESEARCHER:
            raise PolicyError("refusing to drain a researcher GPU")
        draining = set(self.store.get_kv("draining_gpus", []) or [])
        draining.add(uuid)
        self.store.set_kv("draining_gpus", sorted(draining))
        job = None
        if gpu.job_id:
            job = self.preempt(gpu.job_id, reason=f"drain {uuid}")
        self.store.put_gpu(GPUResource.from_dict({**gpu.to_dict(), "occupancy": Occupancy.DRAINING.value}))
        self.emit("GPU_DRAIN", payload={"uuid": uuid, "job_id": gpu.job_id})
        return {"ok": True, "uuid": uuid, "job": job.to_dict() if job else None}

    def undrain_gpu(self, uuid: str) -> dict[str, Any]:
        draining = [u for u in (self.store.get_kv("draining_gpus", []) or []) if u != uuid]
        self.store.set_kv("draining_gpus", draining)
        gpu = next((g for g in self.store.list_gpus() if g.uuid == uuid), None)
        if gpu and gpu.occupancy == Occupancy.DRAINING and not gpu.researcher_present:
            self.store.put_gpu(GPUResource.from_dict({**gpu.to_dict(), "occupancy": Occupancy.AVAILABLE.value}))
        self.emit("GPU_UNDRAIN", payload={"uuid": uuid})
        return {"ok": True, "uuid": uuid}

    def kill_job(self, job_id: str) -> Job:
        job = self.store.get_job(job_id)
        if job.state not in ACTIVE_STATES and job.state != JobState.CHECKPOINTING:
            return job
        self._queue_action(job.worker_id, {"kind": "KILL", "job_id": job.id})
        self.emit("JOB_KILL_REQUESTED", job_id=job.id, run_id=job.run_id, source="ui")
        return job

    def status(self) -> dict[str, Any]:
        # Dashboard polls this even when no worker is heartbeating; reclaim here
        # so Troiani jobs cannot sit RUNNING on a silent worker forever.
        self._reclaim_stale_workers()
        gpus = self.store.list_gpus()
        jobs = self.store.list_jobs()
        now = utcnow()
        stale_s = self.config.policy.worker_stale_s
        workers = []
        for worker in self.store.list_workers():
            data = worker.to_dict()
            age = self._heartbeat_age_s(worker.last_heartbeat)
            data["heartbeat_age_s"] = None if age == float("inf") else age
            data["stale"] = age > stale_s
            workers.append(data)
        refs = {gpu.uuid: gpu_ref(gpu.node, gpu.index, gpu.uuid) for gpu in gpus}
        job_payloads = []
        tokens_total = 0
        for job in jobs:
            payload = job.to_dict()
            payload["runtime_s"] = job.runtime_s(now)
            payload["assigned_refs"] = [refs.get(uuid, uuid) for uuid in job.assigned_gpus]
            if job.run_id:
                try:
                    run = self.store.get_run(job.run_id)
                    payload["val_loss"] = (run.metrics or {}).get("val_loss")
                except NotFound:
                    payload["val_loss"] = None
            tokens_total += job.tokens_ingested
            job_payloads.append(payload)
        snapshot = self.metrics.snapshot()
        snapshot["tokens_ingested"] = max(float(snapshot.get("tokens_ingested") or 0), float(tokens_total))
        intents = self.store.list_intents(200)
        policy = self.policy_dict()
        now_dt = parse_ts(now) or datetime.now(timezone.utc)
        return {
            "now": now,
            "gpus": [g.to_dict() for g in gpus],
            "jobs": job_payloads,
            "workers": workers,
            "runs": [r.to_dict() for r in self.store.list_runs()],
            "intents": [i.to_dict() for i in intents[:50]],
            "annoyance": compute_annoyance(intents, gpus, self.metrics.occupancy_seconds()),
            "user_occupancy": self.metrics.user_occupancy(),
            "events": [e.to_dict() for e in self.store.list_events(80)],
            "metrics": snapshot,
            "policy": policy,
            "policy_open": in_window(self.config.policy, now_dt) and not self.config.policy.stop_all,
            "stop_all": bool(self.config.policy.stop_all),
            "activity_users": self.activity_by_user(),
            "nodes": sorted({gpu.node for gpu in gpus if gpu.node} | set(self.config.nodes)),
            "draining": list(self.store.get_kv("draining_gpus", []) or []),
            "tracking": {
                "dashboard": self.config.control.public_url,
                "mlflow": f"http://127.0.0.1:{self.config.control.mlflow_port}",
                "tensorboard": f"http://127.0.0.1:{self.config.control.tensorboard_port}",
            },
        }

    LOG_CAP = 262144

    def ingest_job_log(self, job_id: str, raw: Any, node: str) -> dict[str, Any]:
        blob = self.store.get_kv(f"log_tail:{job_id}", {}) or {}
        prev_end = int(blob.get("end") or 0)
        text = str(blob.get("text") or "")
        if isinstance(raw, str):
            chunk = raw
            size = len(chunk.encode("utf-8", errors="replace"))
            if not text:
                text = chunk[-self.LOG_CAP :]
            elif chunk not in text:
                text = (text + chunk)[-self.LOG_CAP :]
            end = prev_end + size
        else:
            payload = raw or {}
            chunk = str(payload.get("chunk") or "")
            offset = int(payload.get("offset") or 0)
            size = int(payload.get("size") or len(chunk.encode("utf-8", errors="replace")))
            if offset == 0:
                text = chunk[-self.LOG_CAP :]
                end = size
            elif offset < prev_end:
                return blob if isinstance(blob, dict) else {}
            else:
                text = (text + chunk)[-self.LOG_CAP :]
                end = offset + size
        stored = {"text": text, "ts": utcnow(), "node": node, "end": end}
        self.store.set_kv(f"log_tail:{job_id}", stored)
        return stored

    def get_job_log(self, job_id: str) -> dict[str, Any]:
        self.store.get_job(job_id)
        blob = self.store.get_kv(f"log_tail:{job_id}", {}) or {}
        text = str(blob.get("text") or "")
        source = str(blob.get("node") or "")
        local = self.config.paths.logs / f"{job_id}.log"
        if local.is_file():
            local_text = local.read_text(errors="replace")[-self.LOG_CAP :]
            if not text or len(local_text) > len(text):
                text = local_text
                source = source or "control"
        return {"job_id": job_id, "text": text, "source": source or "none", "ts": blob.get("ts"), "bytes": blob.get("end")}

    def list_templates(self) -> list[dict[str, Any]]:
        jobs_dir = self.config.source.parent / "jobs" if self.config.source else None
        if jobs_dir is None or not jobs_dir.is_dir():
            jobs_dir = Path(__file__).resolve().parents[3] / "config" / "jobs"
        out: list[dict[str, Any]] = []
        if not jobs_dir.is_dir():
            return out
        for path in sorted(jobs_dir.glob("*.yaml")):
            try:
                raw = yaml.safe_load(path.read_text()) or {}
            except (OSError, yaml.YAMLError):
                continue
            if not isinstance(raw, dict):
                continue
            try:
                spec = spec_from_task_yaml(raw)
            except JobSpecError:
                spec = raw if raw.get("command") else None
            if spec:
                out.append({"id": path.stem, "file": path.name, "spec": spec})
        return out

    def activity_by_user(self) -> list[dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for signal in self.store.list_intents(200):
            user = (signal.username or "").strip() or "unknown"
            row = rows.setdefault(
                user,
                {"username": user, "signals": 0, "kinds": {}, "nodes": [], "items": []},
            )
            row["signals"] += 1
            kind = signal.kind.value
            row["kinds"][kind] = int(row["kinds"].get(kind) or 0) + 1
            if signal.node and signal.node not in row["nodes"]:
                row["nodes"].append(signal.node)
            if len(row["items"]) < 8:
                row["items"].append(signal.to_dict())
        occupancy = {f"{u['username']}@{u['node']}": u for u in self.metrics.user_occupancy()}
        out = []
        for row in rows.values():
            seconds = 0.0
            other = 0.0
            for node in row["nodes"] or ["unknown"]:
                occ = occupancy.get(f"{row['username']}@{node}")
                if occ:
                    seconds += float(occ.get("seconds") or 0)
                    other += float(occ.get("other_s") or 0)
            row["seconds"] = seconds
            row["other_s"] = other
            out.append(row)
        out.sort(key=lambda item: (-item["signals"], item["username"]))
        return out

    def infra_report(self) -> dict[str, Any]:
        control = storage_report(self.config.paths)
        nodes = {}
        for worker in self.store.list_workers():
            blob = self.store.get_kv(f"node_infra:{worker.node}") or {}
            if isinstance(blob, dict):
                nodes[worker.node] = blob
        runs = []
        for run in self.store.list_runs():
            ckpts = self.store.list_checkpoints(run.id)
            runs.append(
                {
                    "run_id": run.id,
                    "experiment": run.experiment,
                    "status": run.status,
                    "checkpoints": len(ckpts),
                    "names": [c.id for c in ckpts[-8:]],
                    "logical_path": f"experiments/{run.experiment or 'exp'}/{run.id}",
                }
            )
        return {
            "control": control,
            "nodes": nodes,
            "runs": runs[:40],
            "top": (nodes.get(next(iter(nodes), ""), {}) or {}).get("top") or [],
        }

    def gc_checkpoints(self, run_id: str | None = None) -> dict[str, Any]:
        deleted: dict[str, list[str]] = {}
        targets = [run_id] if run_id else [run.id for run in self.store.list_runs()]
        for rid in targets:
            if not rid:
                continue
            try:
                deleted[rid] = self.checkpoints.gc(rid)
            except CheckpointError:
                deleted[rid] = []
        self.emit("CHECKPOINT_GC", payload={"runs": {k: len(v) for k, v in deleted.items()}})
        return {"ok": True, "deleted": deleted}

    def lineage(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        return {
            "run": run.to_dict(),
            "checkpoints": [c.to_dict() for c in self.store.list_checkpoints(run_id)],
            "artifacts": [a.to_dict() for a in self.store.list_artifacts(run_id)],
            "metrics": self.store.list_metrics(run_id, limit=200),
            "events": [e.to_dict() for e in self.store.list_events(200) if e.run_id == run_id],
        }

    def reproduce(self, run_id: str) -> dict[str, Any]:
        return reproduce_run(self.store.get_run(run_id))

    def validate_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        record = self.store.get_checkpoint(checkpoint_id)
        manifest = self.checkpoints.validate(record.run_id, record.id)
        return {"ok": True, "manifest": manifest}

    def put_artifact(self, artifact: Artifact) -> Artifact:
        return self.store.put_artifact(artifact)

    def export_status_json(self) -> str:
        return json.dumps(self.status(), default=str, indent=2)
