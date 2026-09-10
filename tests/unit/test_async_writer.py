from troiani_platform.models import CheckpointKind
from troiani_platform.training.async_writer import AsyncCheckpointWriter
from troiani_platform.training.checkpoint import CheckpointManager, SaveRequest


def test_named_checkpoint_and_preempt_is_sync(tmp_path):
    mgr = CheckpointManager(tmp_path)
    writer = AsyncCheckpointWriter(mgr)
    writer.submit(
        SaveRequest(
            run_id="run-a",
            job_id="job-a",
            step=10,
            kind=CheckpointKind.REGULAR,
            state={"step": 10},
            experiment="smoke-1gpu",
        )
    )
    writer.wait()
    assert (tmp_path / "run-a" / "smoke-1gpu-ckpt-000010" / "VALID").exists()
    writer.submit(
        SaveRequest(
            run_id="run-a",
            job_id="job-a",
            step=12,
            kind=CheckpointKind.PREEMPTION,
            state={"step": 12},
            experiment="smoke-1gpu",
        )
    )
    assert (tmp_path / "run-a" / "smoke-1gpu-preempt-000012" / "VALID").exists()
    assert (tmp_path / "experiments" / "smoke-1gpu" / "run-a").exists()
