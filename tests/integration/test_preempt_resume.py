from __future__ import annotations

from troiani_platform.models import GPUResource, Occupancy, ProcessInfo
from troiani_platform.simulation.cluster import default_pool
from tests.conftest import make_spec


def _seed_idle(service, gpus=None):
    for gpu in gpus or default_pool():
        service.store.put_gpu(gpu)
    service.store.put_worker(
        __import__("troiani_platform.models", fromlist=["WorkerRecord"]).WorkerRecord(
            id="atlas-w", node="atlas", hostname="atlas"
        )
    )
    service.store.put_worker(
        __import__("troiani_platform.models", fromlist=["WorkerRecord"]).WorkerRecord(
            id="uranus-w", node="uranus", hostname="uranus"
        )
    )


def test_submit_preempt_resume_cycle(service):
    _seed_idle(service)
    job = service.submit(make_spec(name="cycle", gpus=1, min_gpu_memory_gb=40))
    service.tick()
    job = service.get_job(job.id)
    assert job.state.value in {"SCHEDULED", "RESUMING", "STARTING"}
    assert job.assigned_gpus

    claimed = job.assigned_gpus[0]
    gpu = next(g for g in service.store.list_gpus() if g.uuid == claimed)
    service.store.put_gpu(
        GPUResource.from_dict(
            {
                **gpu.to_dict(),
                "occupancy": Occupancy.RESEARCHER.value,
                "compute_processes": [
                    ProcessInfo(
                        pid=7,
                        name="train",
                        gpu_uuid=claimed,
                        memory_used_gb=20,
                        username="gkoutr",
                        troiani_owned=False,
                    ).to_dict()
                ],
            }
        )
    )
    service.tick()
    job = service.get_job(job.id)
    assert job.preempt_requested or job.state.value in {"CHECKPOINTING", "PREEMPTED", "QUEUED"}

    # Pretend trainer finished the emergency checkpoint.
    service.checkpoints.save(
        __import__("troiani_platform.training.checkpoint", fromlist=["SaveRequest"]).SaveRequest(
            run_id=job.run_id,
            job_id=job.id,
            step=42,
            kind=__import__("troiani_platform.models", fromlist=["CheckpointKind"]).CheckpointKind.PREEMPTION,
            state={"step": 42},
            healthy=True,
        )
    )
    service.job_exit({"job_id": job.id, "exit_code": 75})
    job = service.get_job(job.id)
    assert job.state.value in {"QUEUED", "PREEMPTED", "PENDING"}
    assert job.last_step == 42

    # Researcher leaves; job should place again and resume.
    idle = next(g for g in service.store.list_gpus() if g.uuid == claimed)
    service.store.put_gpu(
        GPUResource.from_dict({**idle.to_dict(), "occupancy": Occupancy.AVAILABLE.value, "compute_processes": [], "job_id": None, "memory_used_gb": 0})
    )
    job = service.resume(job.id)
    service.tick()
    job = service.get_job(job.id)
    assert job.state.value in {"SCHEDULED", "RESUMING", "QUEUED"}


def test_resume_twice_is_safe(service):
    _seed_idle(service)
    job = service.submit(make_spec(name="idem"))
    first = service.resume(job.id)
    second = service.resume(job.id)
    assert first.id == second.id


def test_job_exit_keeps_posted_checkpoint_when_files_are_off_node(service):
    _seed_idle(service)
    job = service.submit(make_spec(name="off-node-ckpt"))
    service.tick()
    job = service.get_job(job.id)
    service.record_metrics({"job_id": job.id, "run_id": job.run_id, "step": 55, "loss": 0.4})
    service.record_checkpoint(
        {
            "id": "ckpt-000050",
            "run_id": job.run_id,
            "job_id": job.id,
            "step": 50,
            "kind": "shutdown",
            "healthy": True,
            "checksum": "deadbeef",
            "path": f"/worker/var/checkpoints/{job.run_id}/ckpt-000050",
        }
    )
    assert not list(service.config.paths.checkpoints.glob("*/*"))
    job = service.job_exit({"job_id": job.id, "exit_code": 0, "output_tail": "completed step=55\n"})
    assert job.state.value == "COMPLETED"
    assert job.last_checkpoint_id == "ckpt-000050"
    assert job.last_step == 55
    listed = next(item for item in service.status()["jobs"] if item["id"] == job.id)
    assert listed["last_checkpoint_id"] == "ckpt-000050"


def test_job_exit_completed_text_overrides_nonzero_code(service):
    _seed_idle(service)
    job = service.submit(make_spec(name="done-text"))
    service.tick()
    job = service.get_job(job.id)
    service.job_exit(
        {
            "job_id": job.id,
            "exit_code": 1,
            "output_tail": "completed step=60\n",
        }
    )
    job = service.get_job(job.id)
    assert job.state.value == "COMPLETED"
