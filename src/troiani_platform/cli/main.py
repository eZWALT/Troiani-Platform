from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
import uvicorn
import yaml

from troiani_platform.config import load_config
from troiani_platform.control.api import create_app
from troiani_platform.control.service import PlatformService
from troiani_platform.models import JobSpec
from troiani_platform.worker.worker import Worker

app = typer.Typer(no_args_is_help=True, add_completion=False)
job_app = typer.Typer(no_args_is_help=True)
gpu_app = typer.Typer(no_args_is_help=True)
run_app = typer.Typer(no_args_is_help=True)
ckpt_app = typer.Typer(no_args_is_help=True)
policy_app = typer.Typer(no_args_is_help=True)
app.add_typer(job_app, name="job")
app.add_typer(gpu_app, name="gpu")
app.add_typer(run_app, name="run")
app.add_typer(ckpt_app, name="checkpoint")
app.add_typer(policy_app, name="policy")

platform_app = typer.Typer(no_args_is_help=True, add_completion=False)
troiani_app = typer.Typer(no_args_is_help=True, add_completion=False)
troiani_app.add_typer(app, name="platform")


def _cfg():
    return load_config()


def _client():
    from troiani_platform.cli.client import PlatformClient

    return PlatformClient(_cfg())


def _print(data: object) -> None:
    typer.echo(json.dumps(data, indent=2, default=str))


@app.command("status")
def status() -> None:
    _print(_client().get("/v1/status"))


@gpu_app.command("list")
def gpu_list() -> None:
    _print(_client().get("/v1/gpus"))


@job_app.command("list")
def job_list() -> None:
    _print(_client().get("/v1/jobs"))


@job_app.command("submit")
def job_submit(spec: str) -> None:
    """Submit a Troiani job YAML or a SkyPilot-shaped task file."""
    _print(_client().submit_file(spec))


@job_app.command("pause")
def job_pause(job_id: str) -> None:
    _print(_client().post(f"/v1/jobs/{job_id}/pause"))


@job_app.command("resume")
def job_resume(job_id: str) -> None:
    _print(_client().post(f"/v1/jobs/{job_id}/resume"))


@job_app.command("preempt")
def job_preempt(job_id: str) -> None:
    _print(_client().post(f"/v1/jobs/{job_id}/preempt"))


@job_app.command("cancel")
def job_cancel(job_id: str) -> None:
    _print(_client().post(f"/v1/jobs/{job_id}/cancel"))


@job_app.command("kill")
def job_kill(job_id: str) -> None:
    _print(_client().post(f"/v1/jobs/{job_id}/kill"))


@job_app.command("log")
def job_log(job_id: str) -> None:
    _print(_client().get(f"/v1/jobs/{job_id}/log"))


@job_app.command("templates")
def job_templates() -> None:
    _print(_client().get("/v1/templates"))


@gpu_app.command("drain")
def gpu_drain(uuid: str) -> None:
    _print(_client().post(f"/v1/gpus/{uuid}/drain"))


@gpu_app.command("undrain")
def gpu_undrain(uuid: str) -> None:
    _print(_client().post(f"/v1/gpus/{uuid}/undrain"))


@policy_app.command("show")
def policy_show() -> None:
    _print(_client().get("/v1/policy"))


@policy_app.command("set")
def policy_set(
    max_troiani_gpus: Optional[int] = None,
    max_jobs: Optional[int] = None,
    max_gpus_per_job: Optional[int] = None,
    preempt_grace_s: Optional[int] = None,
    cooldown_s: Optional[int] = None,
    windows: Optional[str] = None,
) -> None:
    patch: dict = {}
    if max_troiani_gpus is not None:
        patch["max_troiani_gpus"] = max_troiani_gpus
    if max_jobs is not None:
        patch["max_jobs"] = max_jobs
    if max_gpus_per_job is not None:
        patch["max_gpus_per_job"] = max_gpus_per_job
    if preempt_grace_s is not None:
        patch["preempt_grace_s"] = preempt_grace_s
    if cooldown_s is not None:
        patch["cooldown_s"] = cooldown_s
    if windows is not None:
        patch["windows"] = [part.strip() for part in windows.split(",") if part.strip()]
    _print(_client().put("/v1/policy", patch))


@app.command("release-all")
def release_all() -> None:
    _print(_client().post("/v1/release-all"))


@run_app.command("inspect")
def run_inspect(run_id: str) -> None:
    _print(_client().get(f"/v1/runs/{run_id}"))


@run_app.command("lineage")
def run_lineage(run_id: str) -> None:
    _print(_client().get(f"/v1/runs/{run_id}/lineage"))


@app.command("reproduce")
def reproduce(run_id: str) -> None:
    _print(_client().get(f"/v1/runs/{run_id}/reproduce"))


@ckpt_app.command("list")
def ckpt_list(run_id: str) -> None:
    data = _client().get(f"/v1/runs/{run_id}/lineage")
    _print(data.get("checkpoints"))


@ckpt_app.command("validate")
def ckpt_validate(checkpoint_id: str) -> None:
    _print(_client().get(f"/v1/checkpoints/{checkpoint_id}/validate"))


@app.command("control")
def control(
    host: Optional[str] = None,
    port: Optional[int] = None,
    with_tracking: bool = False,
) -> None:
    cfg = _cfg()
    if with_tracking or cfg.control.bind_mlflow:
        _spawn_tracking(cfg)
    uvicorn.run(
        create_app(cfg),
        host=host or cfg.control.host,
        port=int(port or cfg.control.port),
        log_level="info",
    )


@app.command("worker")
def worker(node: Optional[str] = None) -> None:
    cfg = _cfg()
    Worker(cfg, node=node).run_forever()


@app.command("run")
def run_cmd(
    gpus: int = typer.Option(1, "--gpus"),
    priority: str = typer.Option("opportunistic"),
    command: list[str] = typer.Argument(None),
) -> None:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise typer.BadParameter("pass a command after --")
    from troiani_platform.models import Priority

    spec = JobSpec(name="adhoc", command=tuple(command), gpus=gpus, priority=Priority(priority))
    spec.validate()
    from troiani_platform.cli.client import PlatformClient

    _print(PlatformClient(_cfg()).post("/v1/jobs", spec.to_dict()))


def _spawn_tracking(cfg) -> None:
    import subprocess
    import sys

    from troiani_platform.tracking.uri import mlflow_artifact_root

    cfg.ensure_dirs()
    logs = Path(cfg.paths.logs)
    mlflow_log = logs / "mlflow.log"
    tb_log = logs / "tensorboard.log"
    mlflow_cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        cfg.mlflow_tracking_uri,
        "--default-artifact-root",
        mlflow_artifact_root(cfg.paths.mlflow),
        "--host",
        "127.0.0.1",
        "--port",
        str(cfg.control.mlflow_port),
    ]
    tb_cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(cfg.paths.tensorboard),
        "--host",
        "127.0.0.1",
        "--port",
        str(cfg.control.tensorboard_port),
    ]
    for cmd, log_path in ((mlflow_cmd, mlflow_log), (tb_cmd, tb_log)):
        try:
            handle = log_path.open("ab")
            subprocess.Popen(cmd, stdout=handle, stderr=handle, env=os.environ.copy())
        except OSError:
            pass

