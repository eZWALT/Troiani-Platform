from fastapi.testclient import TestClient

from troiani_platform.control.api import create_app
from troiani_platform.models import GPUResource, Occupancy, ProcessInfo
from tests.conftest import make_spec


def test_policy_roundtrip(service):
    client = TestClient(create_app(service.config, service))
    shown = client.get("/v1/policy").json()
    assert "max_troiani_gpus" in shown
    names = {row.get("name") for row in shown.get("windows") or []}
    assert "weeknights" in names
    assert shown.get("windows") and all(not row.get("enabled") for row in shown["windows"])
    updated = client.put("/v1/policy", json={"max_troiani_gpus": 2, "windows": ["20:00-08:00"]}).json()
    assert updated["max_troiani_gpus"] == 2
    assert updated["windows"][0]["start"] == "20:00"
    assert updated["windows"][0]["end"] == "08:00"
    named = client.put(
        "/v1/policy",
        json={
            "windows": [
                {
                    "name": "weeknights",
                    "start": "20:00",
                    "end": "08:00",
                    "days": ["mon", "tue", "wed", "thu", "fri"],
                    "enabled": True,
                }
            ]
        },
    ).json()
    assert named["windows"][0]["name"] == "weeknights"
    assert named["windows"][0]["days"] == ["mon", "tue", "wed", "thu", "fri"]


def test_drain_refuses_researcher(service):
    client = TestClient(create_app(service.config, service))
    service.store.put_gpu(
        GPUResource(
            uuid="GPU-research",
            node="atlas",
            index=0,
            name="A100 40GB",
            model="A100",
            memory_gb=40,
            occupancy=Occupancy.RESEARCHER,
            compute_processes=(
                ProcessInfo(pid=1, name="train", gpu_uuid="GPU-research", memory_used_gb=20, username="gkoutr"),
            ),
        )
    )
    denied = client.post("/v1/gpus/GPU-research/drain")
    assert denied.status_code == 409


def test_drain_idle_gpu(service):
    client = TestClient(create_app(service.config, service))
    service.store.put_gpu(
        GPUResource(
            uuid="GPU-idle",
            node="atlas",
            index=1,
            name="A100 40GB",
            model="A100",
            memory_gb=40,
            occupancy=Occupancy.AVAILABLE,
        )
    )
    ok = client.post("/v1/gpus/GPU-idle/drain")
    assert ok.status_code == 200
    assert "GPU-idle" in client.get("/v1/status").json()["draining"]


def test_tokens_and_runtime_on_metrics(service):
    job = service.submit(make_spec(name="tok"))
    service.record_metrics(
        {
            "job_id": job.id,
            "run_id": job.run_id,
            "step": 3,
            "loss": 1.2,
            "tokens_delta": 2048,
            "tokens_ingested": 6144,
            "tokens_per_sec": 1000,
        }
    )
    job = service.get_job(job.id)
    assert job.tokens_ingested == 6144
    assert job.tokens_per_sec == 1000
    status = service.status()
    listed = next(j for j in status["jobs"] if j["id"] == job.id)
    assert listed["tokens_ingested"] == 6144
    assert "runtime_s" in listed


def test_dashboard_assets(service):
    client = TestClient(create_app(service.config, service))
    page = client.get("/")
    assert "/static/app.js" in page.text
    assert "Policy" in page.text
    assert "Launch" in page.text
    assert "window-rows" in page.text
    css = client.get("/static/app.css")
    js = client.get("/static/app.js")
    assert css.status_code == 200
    assert js.status_code == 200
    assert "2 nodes · A100" in page.text
    assert "chart-empty" in page.text
    assert "height: 320px" in css.text
    assert "collectChartSeries" in js.text
    assert 'ctx.fillText("%"' in js.text
    assert 'ctx.fillText("time"' in js.text
    assert "→100B" in page.text
    assert "refreshJobLog" in js.text
    assert "fmtEta100B" in js.text
