from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse

from troiani_platform.config import PlatformConfig, load_config
from troiani_platform.control.service import PlatformService
from pathlib import Path

from fastapi.staticfiles import StaticFiles

from troiani_platform.errors import CheckpointError, JobSpecError, NotFound, PolicyError, TransitionError
from troiani_platform.web.dashboard import render_dashboard


def create_app(config: PlatformConfig | None = None, service: PlatformService | None = None) -> FastAPI:
    cfg = config or load_config()
    svc = service or PlatformService(cfg)
    app = FastAPI(title="Troiani Platform", version="0.1.0")
    app.state.service = svc
    app.state.config = cfg

    def auth(authorization: str | None = Header(default=None)) -> None:
        token = cfg.control.token
        if not token:
            return
        if not authorization or authorization.split(" ", 1)[-1] != token:
            raise HTTPException(status_code=401, detail="unauthorized")

    def require() -> PlatformService:
        return svc

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return render_dashboard(svc.status())

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/status")
    def status(_: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        return service.status()

    @app.get("/v1/gpus")
    def gpus(_: None = Depends(auth), service: PlatformService = Depends(require)) -> list[dict[str, Any]]:
        return service.status()["gpus"]

    @app.get("/v1/jobs")
    def jobs(_: None = Depends(auth), service: PlatformService = Depends(require)) -> list[dict[str, Any]]:
        return service.status()["jobs"]

    @app.post("/v1/jobs")
    def submit(spec: dict[str, Any], _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        try:
            return service.submit(spec).to_dict()
        except JobSpecError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/jobs/from-yaml")
    def submit_yaml(payload: dict[str, Any], _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        try:
            return service.submit_from_yaml(payload).to_dict()
        except JobSpecError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str, _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        try:
            return service.get_job(job_id).to_dict()
        except NotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}/log")
    def get_job_log(job_id: str, _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        try:
            return service.get_job_log(job_id)
        except NotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/templates")
    def templates(_: None = Depends(auth), service: PlatformService = Depends(require)) -> list[dict[str, Any]]:
        return service.list_templates()

    def _job_action(job_id: str, fn) -> dict[str, Any]:
        try:
            return fn(job_id).to_dict()
        except NotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel(job_id: str, _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        return _job_action(job_id, service.cancel)

    @app.post("/v1/jobs/{job_id}/pause")
    def pause(job_id: str, _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        return _job_action(job_id, service.pause)

    @app.post("/v1/jobs/{job_id}/resume")
    def resume(job_id: str, _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        return _job_action(job_id, service.resume)

    @app.post("/v1/jobs/{job_id}/preempt")
    def preempt(job_id: str, _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        return _job_action(job_id, service.preempt)

    @app.post("/v1/release-all")
    def release_all(_: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        jobs = service.release_all()
        return {"released": [j.to_dict() for j in jobs]}

    @app.post("/v1/jobs/{job_id}/kill")
    def kill_job(job_id: str, _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        return _job_action(job_id, service.kill_job)

    @app.get("/v1/policy")
    def get_policy(_: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        return service.policy_dict()

    @app.put("/v1/policy")
    def put_policy(patch: dict[str, Any], _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        try:
            return service.update_policy(patch)
        except PolicyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/policy/stop-all")
    def policy_stop_all(_: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        return service.stop_all()

    @app.post("/v1/policy/resume")
    def policy_resume(_: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        return service.resume_all()

    @app.post("/v1/gpus/{uuid}/drain")
    def drain_gpu(uuid: str, _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        try:
            return service.drain_gpu(uuid)
        except NotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PolicyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/gpus/{uuid}/undrain")
    def undrain_gpu(uuid: str, _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        return service.undrain_gpu(uuid)

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: str, _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        try:
            return service.store.get_run(run_id).to_dict()
        except NotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/runs/{run_id}/lineage")
    def lineage(run_id: str, _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        try:
            return service.lineage(run_id)
        except NotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/runs/{run_id}/reproduce")
    def reproduce(run_id: str, _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        try:
            return service.reproduce(run_id)
        except NotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/checkpoints/{checkpoint_id}/validate")
    def validate_ckpt(checkpoint_id: str, _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        try:
            return service.validate_checkpoint(checkpoint_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/events")
    def events(_: None = Depends(auth), service: PlatformService = Depends(require)) -> list[dict[str, Any]]:
        return [e.to_dict() for e in service.store.list_events(200)]

    @app.get("/v1/activity/users")
    def activity_users(_: None = Depends(auth), service: PlatformService = Depends(require)) -> list[dict[str, Any]]:
        return service.activity_by_user()

    @app.get("/v1/infra")
    def infra(_: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        return service.infra_report()

    @app.post("/v1/infra/gc")
    def infra_gc(payload: dict[str, Any] | None = None, _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        body = payload or {}
        return service.gc_checkpoints(body.get("run_id"))

    @app.get("/v1/metrics")
    def metrics(_: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        return service.metrics.snapshot()

    @app.post("/v1/internal/checkpoint")
    def record_checkpoint(payload: dict[str, Any], _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        try:
            return service.record_checkpoint(payload).to_dict()
        except NotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (CheckpointError, JobSpecError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/internal/heartbeat")
    def heartbeat(payload: dict[str, Any], _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        return service.heartbeat(payload)

    @app.post("/v1/internal/job-exit")
    def job_exit(payload: dict[str, Any], _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, Any]:
        return service.job_exit(payload).to_dict()

    @app.post("/v1/internal/metrics")
    def train_metrics(payload: dict[str, Any], _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, bool]:
        service.record_metrics(payload)
        return {"ok": True}

    @app.post("/v1/internal/anomaly")
    def anomaly(payload: dict[str, Any], _: None = Depends(auth), service: PlatformService = Depends(require)) -> dict[str, bool]:
        service.record_anomaly(payload)
        return {"ok": True}

    static = Path(__file__).resolve().parents[1] / "web" / "static"
    if static.is_dir():
        app.mount("/static", StaticFiles(directory=str(static)), name="static")
    return app
