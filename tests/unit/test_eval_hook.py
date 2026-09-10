import json
import os
import subprocess
import sys

import pytest

from troiani_platform.errors import JobSpecError
from troiani_platform.models import EvaluationHook, Job, JobSpec, Run
from troiani_platform.training import dummy_train
from troiani_platform.training.eval_hook import (
    decode_eval_env,
    due_this_step,
    encode_eval_env,
    run_eval_command,
)
from troiani_platform.training.runner import build_env
from tests.conftest import make_spec


def test_evaluation_command_rejects_shell_string():
    with pytest.raises(JobSpecError, match="list"):
        EvaluationHook.from_dict({"every_steps": 10, "command": "python -m evil"})


def test_run_eval_command_never_uses_shell(monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs

        class _Done:
            returncode = 0
            stdout = json.dumps({"val_loss": 0.5, "ok": True})
            stderr = ""

        return _Done()

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_eval_command(["python", "-m", "troiani_platform.training.dummy_eval"])
    assert seen["kwargs"].get("shell") is False
    assert seen["argv"][0] == sys.executable
    assert isinstance(seen["argv"], list)
    assert result["metrics"]["val_loss"] == 0.5


def test_dummy_eval_writes_metrics(tmp_path):
    result = run_eval_command(
        [sys.executable, "-m", "troiani_platform.training.dummy_eval"],
        env={
            **os.environ,
            "CHECKPOINT_DIR": str(tmp_path),
            "RUN_ID": "run-eval",
            "EVAL_STEP": "20",
        },
    )
    assert result["ok"]
    assert result["metrics"]["val_loss"] == 1.23
    saved = json.loads((tmp_path / "run-eval" / "eval.json").read_text())
    assert saved["step"] == 20


def test_build_env_exports_eval_hook(tmp_path):
    spec = make_spec(
        evaluation=EvaluationHook(
            every_steps=25,
            command=("python", "-m", "troiani_platform.training.dummy_eval"),
        )
    )
    job = Job(id="job-e", spec=spec)
    run = Run(id="run-e", job_id=job.id, experiment="smoke")
    env = build_env(job, run, tmp_path, "http://127.0.0.1:8787")
    decoded = decode_eval_env(env)
    assert decoded == (25, ["python", "-m", "troiani_platform.training.dummy_eval"])
    assert due_this_step(25, 25)
    assert not due_this_step(24, 25)


def test_dummy_train_runs_eval_every_n_steps_and_posts(tmp_path, monkeypatch):
    posted: list[dict] = []
    marker = tmp_path / "evals.log"
    script = tmp_path / "eval_cmd.py"
    script.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "p = Path(os.environ['EVAL_MARKER'])\n"
        "prev = p.read_text() if p.exists() else ''\n"
        "p.write_text(prev + os.environ.get('EVAL_STEP', '') + '\\n')\n"
        "print(json.dumps({'val_loss': 0.42, 'ok': True}))\n"
    )
    monkeypatch.setattr(dummy_train, "_post", lambda _endpoint, path, payload: posted.append({"path": path, **payload}))
    monkeypatch.setattr(dummy_train, "_preempt", False)
    monkeypatch.setattr(dummy_train, "_shutdown", False)
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "ckpts"))
    monkeypatch.setenv("RUN_ID", "run-eval")
    monkeypatch.setenv("JOB_ID", "job-eval")
    monkeypatch.setenv("PLATFORM_ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setenv("CHECKPOINT_EVERY_STEPS", "99")
    monkeypatch.setenv("EVAL_MARKER", str(marker))
    for key, value in encode_eval_env(2, [sys.executable, str(script)]).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, "argv", ["dummy_train", "--steps", "5", "--sleep", "0"])
    with pytest.raises(SystemExit) as exited:
        dummy_train.main()
    assert exited.value.code == 0
    assert marker.read_text().split() == ["2", "4"]
    eval_posts = [item for item in posted if item.get("val_loss") is not None]
    assert [item["step"] for item in eval_posts] == [2, 4]
    assert all(item["path"] == "/v1/internal/metrics" for item in eval_posts)
    assert all(item["val_loss"] == 0.42 for item in eval_posts)


def test_record_metrics_persists_val_loss(service):
    spec = make_spec(
        name="eval-job",
        evaluation=EvaluationHook(every_steps=10, command=("python", "-m", "troiani_platform.training.dummy_eval")),
    )
    job = service.submit(spec)
    service.record_metrics(
        {
            "job_id": job.id,
            "run_id": job.run_id,
            "step": 10,
            "loss": 1.1,
            "val_loss": 0.77,
        }
    )
    names = {row["name"]: row["value"] for row in service.store.list_metrics(job.run_id)}
    assert names["val_loss"] == 0.77
    run = service.store.get_run(job.run_id)
    assert run.metrics["val_loss"] == 0.77


def test_jobspec_evaluation_roundtrip():
    spec = JobSpec.from_dict(
        {
            "name": "with-eval",
            "command": ["python", "-m", "troiani_platform.training.dummy_train"],
            "evaluation": {
                "every_steps": 100,
                "command": ["python", "-m", "troiani_platform.training.dummy_eval"],
            },
        }
    )
    assert spec.evaluation is not None
    assert spec.evaluation.every_steps == 100
    again = JobSpec.from_dict(spec.to_dict())
    assert again.evaluation == spec.evaluation
