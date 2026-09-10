from troiani_platform.models import CheckpointKind
from troiani_platform.training.checkpoint import CheckpointManager, SaveRequest
from troiani_platform.worker.worker import Worker


class _Resp:
    def raise_for_status(self) -> None:
        return None


def test_worker_posts_latest_local_checkpoint(tmp_cfg):
    worker = Worker(tmp_cfg, node="uranus", worker_id="uranus-test")
    posted: list[tuple[str, dict]] = []
    worker.client.post = lambda path, json=None: posted.append((path, json)) or _Resp()
    mgr = CheckpointManager(tmp_cfg.paths.checkpoints)
    mgr.save(
        SaveRequest(
            run_id="run-w",
            job_id="job-w",
            step=12,
            kind=CheckpointKind.SHUTDOWN,
            state={"step": 12},
            healthy=True,
        )
    )
    worker._job_runs["job-w"] = "run-w"
    assert worker._report_latest_checkpoint("job-w")
    assert posted
    path, payload = posted[0]
    assert path == "/v1/internal/checkpoint"
    assert payload["id"] == "ckpt-000012"
    assert payload["run_id"] == "run-w"
    assert payload["job_id"] == "job-w"
    assert payload["step"] == 12
    assert not worker._report_latest_checkpoint("job-w")
