from fastapi.testclient import TestClient

from troiani_platform.control.api import create_app
from tests.conftest import make_spec


def test_submit_and_status(service):
    app = create_app(service.config, service)
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    spec = make_spec().to_dict()
    created = client.post("/v1/jobs", json=spec).json()
    assert created["state"] in {"PENDING", "QUEUED", "SCHEDULED", "HOLD"} or created["id"].startswith("job-")
    listed = client.get("/v1/jobs").json()
    assert listed
    page = client.get("/")
    assert page.status_code == 200
    assert "Troiani Platform" in page.text
    assert "Who" in page.text
    assert "MLflow" in page.text
    assert "2 nodes · A100" in page.text
    assert "STOP ALL" in page.text
    assert "Infra" in page.text
    assert "SM vs VRAM" in page.text or "Solid SM" in page.text


def test_internal_checkpoint_sets_last_checkpoint_without_local_files(service):
    app = create_app(service.config, service)
    client = TestClient(app)
    created = client.post("/v1/jobs", json=make_spec(name="remote-ckpt").to_dict()).json()
    job_id = created["id"]
    run_id = created["run_id"]
    posted = client.post(
        "/v1/internal/checkpoint",
        json={
            "id": "ckpt-000040",
            "run_id": run_id,
            "job_id": job_id,
            "step": 40,
            "kind": "shutdown",
            "healthy": True,
            "checksum": "abc123",
            "path": f"/uranus/var/checkpoints/{run_id}/ckpt-000040",
        },
    )
    assert posted.status_code == 200
    body = posted.json()
    assert body["id"] == "ckpt-000040"
    assert body["path"].startswith("/uranus/")
    assert not (service.config.paths.checkpoints / run_id / "ckpt-000040").exists()

    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["last_checkpoint_id"] == "ckpt-000040"
    assert job["last_step"] == 40

    exited = client.post("/v1/internal/job-exit", json={"job_id": job_id, "exit_code": 0, "success": True})
    assert exited.status_code == 200
    assert exited.json()["last_checkpoint_id"] == "ckpt-000040"
    status = client.get("/v1/status").json()
    listed = next(item for item in status["jobs"] if item["id"] == job_id)
    assert listed["last_checkpoint_id"] == "ckpt-000040"


def test_job_log_and_templates(service):
    app = create_app(service.config, service)
    client = TestClient(app)
    created = client.post("/v1/jobs", json=make_spec(name="logged").to_dict()).json()
    service.store.set_kv(f"log_tail:{created['id']}", {"text": "step 1 loss 1.2\n", "ts": "now", "node": "atlas"})
    log = client.get(f"/v1/jobs/{created['id']}/log").json()
    assert "step 1" in log["text"]
    assert log["source"] == "atlas"
    templates = client.get("/v1/templates").json()
    names = {item["id"] for item in templates}
    assert "smoke-2gpu" in names
    assert "smoke-atlas" in names


def test_incremental_remote_logs_append(service):
    app = create_app(service.config, service)
    client = TestClient(app)
    created = client.post("/v1/jobs", json=make_spec(name="chunks").to_dict()).json()
    job_id = created["id"]
    client.post(
        "/v1/internal/heartbeat",
        json={
            "worker_id": "uranus-w",
            "node": "uranus",
            "gpus": [],
            "jobs": {},
            "logs": {job_id: {"offset": 0, "size": 6, "chunk": "step 1"}},
        },
    )
    client.post(
        "/v1/internal/heartbeat",
        json={
            "worker_id": "uranus-w",
            "node": "uranus",
            "gpus": [],
            "jobs": {},
            "logs": {job_id: {"offset": 6, "size": 7, "chunk": " step 2"}},
        },
    )
    log = client.get(f"/v1/jobs/{job_id}/log").json()
    assert log["text"] == "step 1 step 2"
    assert log["source"] == "uranus"
    assert log["bytes"] == 13
    # replay of an already-acked prefix must not wipe or duplicate
    client.post(
        "/v1/internal/heartbeat",
        json={
            "worker_id": "uranus-w",
            "node": "uranus",
            "gpus": [],
            "jobs": {},
            "logs": {job_id: {"offset": 0, "size": 6, "chunk": "step 1"}},
        },
    )
    assert client.get(f"/v1/jobs/{job_id}/log").json()["text"] == "step 1 step 2"
    client.post(
        "/v1/internal/job-exit",
        json={"job_id": job_id, "exit_code": 0, "success": True, "output_tail": "step 1 step 2\ncompleted step=2"},
    )
    final = client.get(f"/v1/jobs/{job_id}/log").json()
    assert "step 1 step 2" in final["text"]
    assert "completed step=2" in final["text"]
    assert final["text"].count("step 1") == 1


def test_policy_stop_all_and_resume(service):
    app = create_app(service.config, service)
    client = TestClient(app)
    stopped = client.post("/v1/policy/stop-all").json()
    assert stopped["stop_all"] is True
    status = client.get("/v1/status").json()
    assert status["stop_all"] is True
    assert status["policy_open"] is False
    resumed = client.post("/v1/policy/resume").json()
    assert resumed["stop_all"] is False
    assert client.get("/v1/status").json()["policy_open"] is True


def test_from_yaml_and_infra(service):
    app = create_app(service.config, service)
    client = TestClient(app)
    created = client.post(
        "/v1/jobs/from-yaml",
        json={
            "name": "sky-smoke",
            "resources": {"accelerators": "A100:1"},
            "run": "python -m troiani_platform.training.dummy_train --steps 4",
        },
    )
    assert created.status_code == 200
    body = created.json()
    assert body["spec"]["command"][0] == "python"
    assert body["spec"]["gpus"] == 1
    infra = client.get("/v1/infra").json()
    assert "control" in infra
    assert "folders" in infra["control"]
    users = client.get("/v1/activity/users").json()
    assert isinstance(users, list)


def test_internal_checkpoint_rejects_incomplete_manifest(service):
    app = create_app(service.config, service)
    client = TestClient(app)
    response = client.post("/v1/internal/checkpoint", json={"step": 1, "healthy": True})
    assert response.status_code == 400
