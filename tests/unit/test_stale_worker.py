from datetime import datetime, timedelta, timezone

import pytest

from troiani_platform.models import (
    GPUResource,
    Job,
    JobState,
    Occupancy,
    ProcessInfo,
    Run,
    WorkerRecord,
    utcnow,
)
from troiani_platform.worker.watchdog import Watchdog
from tests.conftest import make_spec


def _old_heartbeat(service, extra_s: float = 5.0) -> str:
    age = service.config.policy.worker_stale_s + extra_s
    return (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat()


def _put_worker(service, worker_id: str, node: str, last_heartbeat: str, status: str = "alive") -> None:
    service.store.put_worker(
        WorkerRecord(
            id=worker_id,
            node=node,
            hostname=node,
            last_heartbeat=last_heartbeat,
            status=status,
        )
    )


def _put_gpu(
    service,
    uuid: str,
    node: str,
    *,
    job_id: str | None = None,
    worker_id: str | None = None,
    occupancy: Occupancy = Occupancy.TROIANI,
    processes: tuple[ProcessInfo, ...] = (),
) -> GPUResource:
    gpu = GPUResource(
        uuid=uuid,
        node=node,
        index=0,
        name="A100 40GB",
        model="A100",
        memory_gb=40,
        occupancy=occupancy,
        job_id=job_id,
        worker_id=worker_id,
        compute_processes=processes,
    )
    service.store.put_gpu(gpu)
    return gpu


def _put_job(
    service,
    *,
    state: JobState = JobState.RUNNING,
    worker_id: str = "atlas-w",
    node: str = "atlas",
    gpu: str = "gpu-atlas-0",
    job_id: str = "job-stale",
    pid: int | None = 4242,
) -> Job:
    job = Job(
        id=job_id,
        spec=make_spec(name=job_id),
        state=state,
        run_id=f"run-{job_id}",
        assigned_gpus=[gpu],
        assigned_node=node,
        worker_id=worker_id,
        started_at=utcnow(),
        pid=pid,
    )
    service.store.put_job(job)
    if job.run_id:
        service.store.put_run(Run(id=job.run_id, job_id=job.id, experiment=job.spec.name))
    return job


def _action_kinds(service) -> list[str]:
    return [action.get("kind") for queued in service._pending_actions.values() for action in queued]


def test_stale_worker_requeues_running_job(service):
    job = _put_job(service)
    _put_gpu(service, "gpu-atlas-0", "atlas", job_id=job.id, worker_id="atlas-w")
    _put_worker(service, "atlas-w", "atlas", _old_heartbeat(service))
    service._queue_action("atlas-w", {"kind": "START", "job_id": job.id})

    service.tick()

    job = service.get_job(job.id)
    gpu = next(g for g in service.store.list_gpus() if g.uuid == "gpu-atlas-0")
    events = [event for event in service.store.list_events() if event.type == "WORKER_STALE"]
    worker = next(w for w in service.store.list_workers() if w.id == "atlas-w")

    assert job.state is JobState.QUEUED
    assert job.assigned_gpus == []
    assert job.assigned_node is None
    assert job.worker_id is None
    assert gpu.job_id is None
    assert gpu.occupancy is Occupancy.AVAILABLE
    assert events
    assert events[0].payload.get("job_ids") == [job.id]
    assert worker.status == "stale"
    assert "KILL" not in _action_kinds(service)
    assert service._pending_actions.get("atlas-w") in (None, [])


def test_status_reclaims_without_heartbeat_tick(service):
    job = _put_job(service)
    _put_gpu(service, "gpu-atlas-0", "atlas", job_id=job.id, worker_id="atlas-w")
    _put_worker(service, "atlas-w", "atlas", _old_heartbeat(service))

    snapshot = service.status()

    job = service.get_job(job.id)
    listed = next(item for item in snapshot["jobs"] if item["id"] == job.id)
    worker = next(item for item in snapshot["workers"] if item["id"] == "atlas-w")
    assert job.state is JobState.QUEUED
    assert listed["state"] == JobState.QUEUED.value
    assert worker["stale"] is True
    assert any(event["type"] == "WORKER_STALE" for event in snapshot["events"])


def test_fresh_worker_keeps_running_job(service):
    job = _put_job(service)
    _put_gpu(service, "gpu-atlas-0", "atlas", job_id=job.id, worker_id="atlas-w")
    _put_worker(service, "atlas-w", "atlas", utcnow())

    service.tick()

    job = service.get_job(job.id)
    gpu = next(g for g in service.store.list_gpus() if g.uuid == "gpu-atlas-0")
    assert job.state is JobState.RUNNING
    assert job.assigned_gpus == ["gpu-atlas-0"]
    assert gpu.job_id == job.id
    assert not any(event.type == "WORKER_STALE" for event in service.store.list_events())


@pytest.mark.parametrize(
    "state",
    [
        JobState.SCHEDULED,
        JobState.STARTING,
        JobState.RUNNING,
        JobState.CHECKPOINTING,
        JobState.RESUMING,
    ],
)
def test_stale_reclaim_uses_valid_transitions(service, state):
    job = _put_job(service, state=state, job_id=f"job-{state.value.lower()}")
    _put_gpu(service, "gpu-atlas-0", "atlas", job_id=job.id, worker_id="atlas-w")
    _put_worker(service, "atlas-w", "atlas", _old_heartbeat(service))

    service.tick()

    job = service.get_job(job.id)
    assert job.state is JobState.QUEUED


def test_stale_worker_does_not_affect_live_worker_jobs(service):
    live = _put_job(service, job_id="job-live", worker_id="uranus-w", node="uranus", gpu="gpu-uranus-0")
    dead = _put_job(service, job_id="job-dead", worker_id="atlas-w", node="atlas", gpu="gpu-atlas-0")
    _put_gpu(service, "gpu-uranus-0", "uranus", job_id=live.id, worker_id="uranus-w")
    _put_gpu(service, "gpu-atlas-0", "atlas", job_id=dead.id, worker_id="atlas-w")
    _put_worker(service, "uranus-w", "uranus", utcnow())
    _put_worker(service, "atlas-w", "atlas", _old_heartbeat(service))

    service.tick()

    live = service.get_job(live.id)
    dead = service.get_job(dead.id)
    assert live.state is JobState.RUNNING
    assert live.assigned_gpus == ["gpu-uranus-0"]
    assert dead.state is JobState.QUEUED
    assert dead.assigned_gpus == []


def test_stale_node_is_not_rescheduled_until_heartbeat(service):
    job = _put_job(service)
    _put_gpu(service, "gpu-atlas-0", "atlas", job_id=job.id, worker_id="atlas-w")
    _put_worker(service, "atlas-w", "atlas", _old_heartbeat(service))

    service.tick()
    job = service.get_job(job.id)
    assert job.state is JobState.QUEUED
    assert job.assigned_gpus == []

    service.tick()
    job = service.get_job(job.id)
    assert job.state is JobState.QUEUED
    assert job.assigned_gpus == []
    stale_events = [event for event in service.store.list_events() if event.type == "WORKER_STALE"]
    assert len(stale_events) == 1


def test_stale_reclaim_moves_job_to_live_node(service):
    job = _put_job(service)
    _put_gpu(service, "gpu-atlas-0", "atlas", job_id=job.id, worker_id="atlas-w")
    _put_gpu(service, "gpu-uranus-0", "uranus", occupancy=Occupancy.AVAILABLE)
    _put_worker(service, "atlas-w", "atlas", _old_heartbeat(service))
    _put_worker(service, "uranus-w", "uranus", utcnow())

    service.tick()

    job = service.get_job(job.id)
    assert job.state is JobState.SCHEDULED
    assert job.assigned_node == "uranus"
    assert job.assigned_gpus == ["gpu-uranus-0"]
    atlas = next(g for g in service.store.list_gpus() if g.uuid == "gpu-atlas-0")
    assert atlas.job_id is None


def test_stale_reclaim_does_not_invent_researcher_kills(service):
    researcher = ProcessInfo(
        pid=7,
        name="vllm",
        gpu_uuid="gpu-atlas-0",
        memory_used_gb=20,
        username="gkoutr",
        troiani_owned=False,
        command="vllm serve",
    )
    job = _put_job(service, pid=4242)
    _put_gpu(
        service,
        "gpu-atlas-0",
        "atlas",
        job_id=job.id,
        worker_id="atlas-w",
        occupancy=Occupancy.RESEARCHER,
        processes=(researcher,),
    )
    _put_worker(service, "atlas-w", "atlas", _old_heartbeat(service))

    service.tick()

    job = service.get_job(job.id)
    gpu = next(g for g in service.store.list_gpus() if g.uuid == "gpu-atlas-0")
    assert job.state is JobState.QUEUED
    assert gpu.job_id is None
    assert gpu.occupancy is Occupancy.RESEARCHER
    assert gpu.compute_processes[0].pid == 7
    assert gpu.compute_processes[0].username == "gkoutr"
    assert "KILL" not in _action_kinds(service)
    assert "STOP" not in _action_kinds(service)
    assert "PREEMPT" not in _action_kinds(service)


def test_pending_placeholder_job_reclaimed_when_node_worker_stale(service):
    job = _put_job(service, worker_id="atlas-pending")
    _put_gpu(service, "gpu-atlas-0", "atlas", job_id=job.id, worker_id="atlas-pending")
    _put_worker(service, "atlas-w", "atlas", _old_heartbeat(service))

    service.tick()

    job = service.get_job(job.id)
    assert job.state is JobState.QUEUED
    assert job.assigned_gpus == []


def test_heartbeat_dead_process_completes_finished_preempt(service):
    job = _put_job(service, state=JobState.CHECKPOINTING, pid=None)
    job = job.touch(preempt_requested=True, last_step=50)
    service.store.put_job(job)
    _put_gpu(service, "gpu-atlas-0", "atlas", job_id=job.id, worker_id="atlas-w")
    _put_worker(service, "atlas-w", "atlas", utcnow())

    service.heartbeat(
        {
            "worker_id": "atlas-w",
            "node": "atlas",
            "gpus": [],
            "jobs": {job.id: {"pid": 4242, "alive": False, "output_tail": "completed step=50"}},
        }
    )

    job = service.get_job(job.id)
    assert job.state is JobState.COMPLETED
    assert job.assigned_gpus == []


def test_heartbeat_dead_process_requeues_unfinished_preempt(service):
    job = _put_job(service, state=JobState.CHECKPOINTING)
    job = job.touch(preempt_requested=True, last_step=20)
    service.store.put_job(job)
    _put_gpu(service, "gpu-atlas-0", "atlas", job_id=job.id, worker_id="atlas-w")
    _put_worker(service, "atlas-w", "atlas", utcnow())

    service.heartbeat(
        {
            "worker_id": "atlas-w",
            "node": "atlas",
            "gpus": [],
            "jobs": {job.id: {"pid": 4242, "alive": False, "output_tail": "preempted at step=20"}},
        }
    )

    job = service.get_job(job.id)
    assert job.state is not JobState.CHECKPOINTING
    assert job.state in {JobState.QUEUED, JobState.SCHEDULED, JobState.RESUMING}


def test_heartbeat_does_not_fail_scheduled_job_before_start(service):
    job = _put_job(service, state=JobState.SCHEDULED)
    _put_gpu(service, "gpu-atlas-0", "atlas", job_id=job.id, worker_id="atlas-w")
    _put_worker(service, "atlas-w", "atlas", utcnow())

    service.heartbeat({"worker_id": "atlas-w", "node": "atlas", "gpus": [], "jobs": {}})

    job = service.get_job(job.id)
    assert job.state is JobState.SCHEDULED


def test_watchdog_control_lost_release_still_honors_grace():
    dog = Watchdog(heartbeat_s=1, stale_s=5, control_lost_grace_s=10)
    dog.mark_control(True)
    assert not dog.control_lost(now=dog.last_control_ok + 9)
    assert dog.control_lost(now=dog.last_control_ok + 11)
