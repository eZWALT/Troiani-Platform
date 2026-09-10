from troiani_platform.models import AnomalyKind, CheckpointKind, CheckpointRecord, CheckpointState
from troiani_platform.training.health import HealthMonitor
from tests.conftest import make_spec
from tests.integration.test_preempt_resume import _seed_idle


def test_nan_pauses_without_forced_kill(service):
    _seed_idle(service)
    job = service.submit(make_spec(name="nanjob"))
    service.record_metrics({"job_id": job.id, "run_id": job.run_id, "step": 3, "loss": 1.2})
    service.record_anomaly({"job_id": job.id, "run_id": job.run_id, "kinds": [AnomalyKind.LOSS_NAN.value]})
    job = service.get_job(job.id)
    assert job.state.value in {"ANOMALY", "PAUSED", "QUEUED"}


def test_fallback_latest_valid(service):
    _seed_idle(service)
    job = service.submit(make_spec(name="fb"))
    good = CheckpointRecord(
        id="ckpt-000010",
        run_id=job.run_id,
        job_id=job.id,
        step=10,
        kind=CheckpointKind.REGULAR,
        state=CheckpointState.VALID,
        healthy=True,
    )
    bad = CheckpointRecord(
        id="ckpt-000020",
        run_id=job.run_id,
        job_id=job.id,
        step=20,
        kind=CheckpointKind.REGULAR,
        state=CheckpointState.CORRUPTED,
    )
    service.store.put_checkpoint(good)
    service.store.put_checkpoint(bad)
    rolled = service.rollback(job.id)
    assert rolled.last_checkpoint_id == "ckpt-000010"


def test_health_monitor_matches_spec():
    mon = HealthMonitor()
    assert mon.observe(ts=1, loss=float("nan"), step=1)
