from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx

from troiani_platform.config import PlatformConfig
from troiani_platform.discovery.gpu import discover_gpus
from troiani_platform.discovery.nodes import current_node, hostname
from troiani_platform.discovery.processes import list_logins
from troiani_platform.log import get_logger
from troiani_platform.models import IntentKind, IntentSignal, Occupancy, utcnow
from troiani_platform.infra.storage import node_snapshot
from troiani_platform.training.checkpoint import CheckpointManager, manifest_to_report
from troiani_platform.worker.executor import ProcessExecutor
from troiani_platform.worker.lifecycle import request_checkpoint, terminate
from troiani_platform.worker.watchdog import Watchdog

log = get_logger("troiani.platform.worker")


class Worker:
    def __init__(self, config: PlatformConfig, node: str | None = None, worker_id: str | None = None) -> None:
        self.config = config
        self.node = current_node(node)
        self.worker_id = worker_id or f"{self.node}-{hostname()}"
        self.client = httpx.Client(
            base_url=config.control.public_url.rstrip("/"),
            timeout=10.0,
            headers=self._headers(),
        )
        self.executor = ProcessExecutor()
        self.watchdog = Watchdog(
            heartbeat_s=config.policy.worker_heartbeat_s,
            stale_s=config.policy.worker_stale_s,
            control_lost_grace_s=config.policy.control_lost_grace_s,
        )
        self.procs: dict[str, Any] = {}
        self._job_runs: dict[str, str] = {}
        self._reported_checkpoints: dict[str, str] = {}
        self._log_offsets: dict[str, int] = {}
        self._pending_logs: dict[str, dict[str, Any]] = {}
        self._log_paths: dict[str, Path] = {}
        self._stop = False

    def _headers(self) -> dict[str, str]:
        token = self.config.control.token
        headers = {"X-Troiani-Worker": self.worker_id, "X-Troiani-Node": self.node}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def snapshot(self) -> list[dict[str, Any]]:
        gpus = discover_gpus(
            node=self.node,
            owned_users=self.config.owned_users,
            memory_busy_gb=self.config.policy.memory_busy_gb,
        )
        return [g.to_dict() for g in gpus]

    def intents(self) -> list[dict[str, Any]]:
        signals: list[IntentSignal] = []
        for session in list_logins():
            user = session.get("user")
            if user and user not in self.config.owned_users:
                signals.append(
                    IntentSignal(
                        kind=IntentKind.LOGIN,
                        node=self.node,
                        username=user,
                        detail=session.get("raw", ""),
                    )
                )
        for gpu in discover_gpus(node=self.node, owned_users=self.config.owned_users):
            if gpu.occupancy == Occupancy.RESEARCHER:
                user = gpu.compute_processes[0].username if gpu.compute_processes else None
                signals.append(
                    IntentSignal(
                        kind=IntentKind.FOREIGN_PROCESS,
                        node=self.node,
                        gpu_uuid=gpu.uuid,
                        username=user,
                        detail=gpu.compute_processes[0].command if gpu.compute_processes else "busy memory",
                    )
                )
        return [s.to_dict() for s in signals]

    def _read_log_delta(self, job_id: str, path: Any) -> dict[str, Any] | None:
        if not path:
            return None
        log_path = Path(path)
        if not log_path.exists():
            return None
        try:
            data = log_path.read_bytes()
        except OSError:
            return None
        offset = int(self._log_offsets.get(job_id, 0))
        if offset > len(data):
            offset = 0
        chunk = data[offset : offset + 32768]
        if not chunk:
            return None
        return {
            "offset": offset,
            "size": len(chunk),
            "chunk": chunk.decode("utf-8", errors="replace"),
        }

    def _collect_logs(self) -> dict[str, dict[str, Any]]:
        logs: dict[str, dict[str, Any]] = {}
        paths = dict(self._log_paths)
        for job_id, proc in self.procs.items():
            raw = getattr(proc, "_troiani_log_path", None)
            if raw:
                paths[job_id] = Path(raw)
                self._log_paths[job_id] = Path(raw)
        for job_id, path in paths.items():
            item = self._read_log_delta(job_id, path)
            if item:
                logs[job_id] = item
        for job_id, item in self._pending_logs.items():
            logs.setdefault(job_id, item)
        return logs

    def _commit_logs(self, logs: dict[str, dict[str, Any]]) -> None:
        for job_id, item in logs.items():
            self._log_offsets[job_id] = int(item.get("offset") or 0) + int(item.get("size") or 0)
            self._pending_logs.pop(job_id, None)
            path = self._log_paths.get(job_id)
            if path is None or job_id in self.procs:
                continue
            try:
                if self._log_offsets[job_id] >= Path(path).stat().st_size:
                    self._log_paths.pop(job_id, None)
            except OSError:
                self._log_paths.pop(job_id, None)

    def heartbeat(self) -> dict[str, Any]:
        logs = self._collect_logs()
        payload = {
            "worker_id": self.worker_id,
            "node": self.node,
            "hostname": hostname(),
            "gpus": self.snapshot(),
            "intents": self.intents(),
            "jobs": {
                job_id: {"pid": proc.pid, "alive": proc.poll() is None}
                for job_id, proc in self.procs.items()
            },
            "logs": logs,
            "infra": node_snapshot(self.config.paths),
            "ts": utcnow(),
        }
        try:
            response = self.client.post("/v1/internal/heartbeat", json=payload)
            response.raise_for_status()
            self.watchdog.mark_control(True)
            self.watchdog.beat()
            self._commit_logs(logs)
            return response.json()
        except Exception as exc:
            self.watchdog.mark_control(False)
            log.warning("heartbeat failed", extra={"event": "heartbeat_failed", "worker_id": self.worker_id})
            return {"ok": False, "error": str(exc), "actions": []}

    def apply_actions(self, actions: list[dict[str, Any]]) -> None:
        for action in actions:
            try:
                kind = action.get("kind")
                job = action.get("job") or {}
                run = action.get("run") or {}
                job_id = action.get("job_id") or job.get("id")
                proc = self.procs.get(job_id) if job_id else None
                alive = proc is not None and proc.poll() is None
                if kind == "START":
                    self._start(job, run, action.get("local_indexes") or [])
                elif kind == "PREEMPT" and alive:
                    request_checkpoint(proc.pid)
                elif kind == "STOP" and alive:
                    terminate(proc.pid, graceful=True)
                elif kind == "KILL" and alive:
                    terminate(proc.pid, graceful=False)
            except OSError:
                log.warning("action failed", extra={"kind": action.get("kind"), "job_id": action.get("job_id")})

    def _start(self, job_raw: dict[str, Any], run_raw: dict[str, Any], local_indexes: list[int]) -> None:
        from troiani_platform.models import Job, Run

        job = Job.from_dict(job_raw)
        run = Run.from_dict(run_raw)
        if job.id in self.procs and self.procs[job.id].poll() is None:
            return
        proc = self.executor.start(
            job,
            run,
            self.config.paths.checkpoints,
            self.config.control.public_url,
            self.config.control.token,
            local_indexes,
        )
        self.procs[job.id] = proc
        self._job_runs[job.id] = run.id
        log_path = getattr(proc, "_troiani_log_path", None)
        if log_path:
            self._log_paths[job.id] = Path(log_path)
        log.info("started job", extra={"job_id": job.id, "run_id": run.id, "worker_id": self.worker_id})

    def _report_latest_checkpoint(self, job_id: str, run_id: str | None = None) -> bool:
        run_id = run_id or self._job_runs.get(job_id)
        if not run_id:
            return False
        latest = CheckpointManager(self.config.paths.checkpoints).latest_valid(run_id)
        if not latest:
            return False
        ckpt_id = str((latest.get("checkpoint") or {}).get("id") or "")
        if not ckpt_id or self._reported_checkpoints.get(job_id) == ckpt_id:
            return False
        path = self.config.paths.checkpoints / run_id / ckpt_id
        try:
            response = self.client.post("/v1/internal/checkpoint", json=manifest_to_report(latest, path))
            response.raise_for_status()
            self._reported_checkpoints[job_id] = ckpt_id
            return True
        except Exception:
            log.warning("checkpoint report failed", extra={"job_id": job_id, "run_id": run_id})
            return False

    def reap(self) -> list[dict[str, Any]]:
        finished = []
        for job_id, proc in list(self.procs.items()):
            code = proc.poll()
            if code is None:
                continue
            log_path = getattr(proc, "_troiani_log_path", None)
            handle = getattr(proc, "_troiani_log", None)
            if handle:
                try:
                    handle.close()
                except OSError:
                    pass
            output = ""
            if log_path and Path(log_path).exists():
                self._log_paths[job_id] = Path(log_path)
                output = Path(log_path).read_text(errors="replace")[-4000:]
                leftover = self._read_log_delta(job_id, log_path)
                if leftover:
                    self._pending_logs[job_id] = leftover
            finished.append({"job_id": job_id, "exit_code": code, "output_tail": output})
            self._report_latest_checkpoint(job_id)
            del self.procs[job_id]
            self._job_runs.pop(job_id, None)
            self._reported_checkpoints.pop(job_id, None)
            try:
                self.client.post("/v1/internal/job-exit", json=finished[-1])
            except Exception:
                log.warning("job-exit report failed", extra={"job_id": job_id})
        return finished

    def safe_release_if_partitioned(self) -> None:
        if not self.watchdog.control_lost():
            return
        log.warning("control plane lost; releasing jobs", extra={"worker_id": self.worker_id, "event": "control_lost"})
        for job_id, proc in list(self.procs.items()):
            if proc.poll() is None:
                try:
                    request_checkpoint(proc.pid)
                    time.sleep(min(5, self.config.policy.preempt_grace_s))
                    if proc.poll() is None:
                        terminate(proc.pid, graceful=True)
                except OSError:
                    pass
            self.procs.pop(job_id, None)

    def run_forever(self) -> None:
        log.info("worker starting", extra={"worker_id": self.worker_id, "node": self.node})
        while not self._stop:
            try:
                reply = self.heartbeat()
                self.apply_actions(reply.get("actions") or [])
                for running_id in list(self.procs):
                    self._report_latest_checkpoint(running_id)
                self.reap()
                self.safe_release_if_partitioned()
            except Exception:
                log.exception("worker loop error", extra={"worker_id": self.worker_id})
            time.sleep(self.config.policy.worker_heartbeat_s)

    def stop(self) -> None:
        self._stop = True
