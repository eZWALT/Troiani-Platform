from pathlib import Path

from fastapi.testclient import TestClient

from troiani_platform.control.api import create_app
from troiani_platform.worker.worker import Worker
from tests.conftest import make_spec


class _Ok:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"ok": True, "actions": []}


def test_incremental_remote_logs_append(service):
    client = TestClient(create_app(service.config, service))
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


def test_duplicate_offset_does_not_rewrite(service):
    client = TestClient(create_app(service.config, service))
    created = client.post("/v1/jobs", json=make_spec(name="retry").to_dict()).json()
    job_id = created["id"]
    first = {"offset": 0, "size": 6, "chunk": "step 1"}
    client.post("/v1/internal/heartbeat", json={"worker_id": "u", "node": "uranus", "gpus": [], "jobs": {}, "logs": {job_id: first}})
    client.post("/v1/internal/heartbeat", json={"worker_id": "u", "node": "uranus", "gpus": [], "jobs": {}, "logs": {job_id: first}})
    assert client.get(f"/v1/jobs/{job_id}/log").json()["text"] == "step 1"


def test_job_exit_tail_fills_empty_log(service):
    client = TestClient(create_app(service.config, service))
    created = client.post("/v1/jobs", json=make_spec(name="exit-log").to_dict()).json()
    job_id = created["id"]
    client.post("/v1/internal/job-exit", json={"job_id": job_id, "exit_code": 0, "success": True, "output_tail": "completed step=4\n"})
    log = client.get(f"/v1/jobs/{job_id}/log").json()
    assert "completed step=4" in log["text"]


def test_failed_heartbeat_does_not_advance_log_offset(tmp_cfg, tmp_path):
    worker = Worker(tmp_cfg, node="uranus", worker_id="u-log")
    log_path = tmp_path / "job-a.log"
    log_path.write_bytes(b"hello\n")

    class Proc:
        pid = 1
        _troiani_log_path = log_path

        def poll(self):
            return None

    worker.procs["job-a"] = Proc()
    worker.snapshot = lambda: []
    worker.intents = lambda: []

    def fail(*_args, **_kwargs):
        raise RuntimeError("control down")

    worker.client.post = fail
    worker.heartbeat()
    assert worker._log_offsets.get("job-a", 0) == 0

    posted: list[dict] = []

    def ok(_path, json=None):
        posted.append(json or {})
        return _Ok()

    worker.client.post = ok
    worker.heartbeat()
    assert posted[0]["logs"]["job-a"]["chunk"].startswith("hello")
    first_end = worker._log_offsets["job-a"]
    assert first_end == 6

    log_path.write_bytes(b"hello\nworld\n")
    posted.clear()
    worker.heartbeat()
    chunk = posted[0]["logs"]["job-a"]
    assert chunk["offset"] == 6
    assert "world" in chunk["chunk"]
    assert worker._log_offsets["job-a"] > first_end


def test_worker_ships_next_chunk_after_commit(tmp_cfg, tmp_path):
    worker = Worker(tmp_cfg, node="atlas", worker_id="a-log")
    log_path = Path(tmp_path / "job-b.log")
    log_path.write_bytes(b"# start\nstep=1\n")

    class Proc:
        pid = 1
        _troiani_log_path = log_path

        def poll(self):
            return None

    worker.procs["job-b"] = Proc()
    logs = worker._collect_logs()
    assert "job-b" in logs
    worker._commit_logs(logs)
    log_path.write_bytes(b"# start\nstep=1\nstep=2\n")
    again = worker._collect_logs()
    assert "step=2" in again["job-b"]["chunk"]
    assert again["job-b"]["offset"] > 0
