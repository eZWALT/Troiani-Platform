import json

import pytest

from troiani_platform.errors import CheckpointError
from troiani_platform.models import CheckpointKind
from troiani_platform.training.checkpoint import CheckpointManager, SaveRequest, manifest_to_report


def _save(mgr: CheckpointManager, step: int, **kwargs):
    return mgr.save(
        SaveRequest(
            run_id="run-1",
            job_id="job-1",
            step=step,
            kind=kwargs.get("kind", CheckpointKind.REGULAR),
            state={"step": step, "w": [1.0, 2.0]},
            healthy=kwargs.get("healthy", True),
            metric=kwargs.get("metric"),
        )
    )


def test_atomic_valid_checkpoint(tmp_path):
    mgr = CheckpointManager(tmp_path)
    manifest = _save(mgr, 10)
    assert (tmp_path / "run-1" / "ckpt-000010" / "VALID").exists()
    assert manifest["checkpoint"]["step"] == 10
    loaded = mgr.load("run-1")
    assert loaded["step"] == 10


def test_partial_write_not_valid(tmp_path):
    mgr = CheckpointManager(tmp_path)
    mgr.faults.add("partial_write")
    with pytest.raises(CheckpointError):
        _save(mgr, 3)
    assert not (tmp_path / "run-1" / "ckpt-000003" / "VALID").exists()
    assert mgr.latest_valid("run-1") is None


def test_corruption_fallback(tmp_path):
    mgr = CheckpointManager(tmp_path)
    _save(mgr, 100)
    _save(mgr, 150)
    bad = tmp_path / "run-1" / "ckpt-000150" / "state.pkl"
    bad.write_bytes(b"not-a-checkpoint")
    latest = mgr.latest_valid("run-1")
    assert latest["checkpoint"]["step"] == 100


def test_manifest_to_report_flattens_worker_save(tmp_path):
    mgr = CheckpointManager(tmp_path)
    manifest = _save(mgr, 7)
    path = tmp_path / "run-1" / "ckpt-000007"
    report = manifest_to_report(manifest, path)
    assert report["id"] == "ckpt-000007"
    assert report["run_id"] == "run-1"
    assert report["job_id"] == "job-1"
    assert report["step"] == 7
    assert report["kind"] == "regular"
    assert report["healthy"] is True
    assert report["checksum"]
    assert report["path"] == str(path)


def test_retention_keeps_recovery(tmp_path):
    mgr = CheckpointManager(tmp_path)
    for step in range(1, 12):
        _save(mgr, step)
    remaining = [p.name for p in (tmp_path / "run-1").iterdir() if p.is_dir() and not p.name.endswith(".tmp")]
    assert "ckpt-000011" in remaining
    assert len(remaining) <= 8
