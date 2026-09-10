import sys

import pytest

from troiani_platform.training import dummy_train


def test_dummy_train_posts_checkpoint_after_save(tmp_path, monkeypatch, capsys):
    posted: list[tuple[str, dict]] = []

    def capture(_endpoint: str, path: str, payload: dict) -> None:
        posted.append((path, payload))

    dummy_train._preempt = False
    dummy_train._shutdown = False
    monkeypatch.setattr(dummy_train, "_post", capture)
    monkeypatch.setenv("RUN_ID", "run-remote")
    monkeypatch.setenv("JOB_ID", "job-remote")
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setenv("PLATFORM_ENDPOINT", "http://control.example")
    monkeypatch.setenv("CHECKPOINT_EVERY_STEPS", "2")
    monkeypatch.setattr(sys, "argv", ["dummy_train", "--steps", "2", "--sleep", "0"])

    with pytest.raises(SystemExit) as exc:
        dummy_train.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "step=1" in out
    assert "step=2" in out

    ckpt_posts = [payload for path, payload in posted if path == "/v1/internal/checkpoint"]
    assert ckpt_posts
    last = ckpt_posts[-1]
    assert last["id"]
    assert last["run_id"] == "run-remote"
    assert last["job_id"] == "job-remote"
    assert last["step"] == 2
    assert last["healthy"] is True
    assert last["path"]
