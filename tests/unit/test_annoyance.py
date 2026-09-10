from fastapi.testclient import TestClient

from troiani_platform.control.api import create_app
from troiani_platform.control.service import PlatformService
from troiani_platform.models import GPUResource, IntentKind, IntentSignal, Occupancy, ProcessInfo
from troiani_platform.monitoring.annoyance import compute_annoyance
from troiani_platform.monitoring.metrics import InfraMetrics


def _gpu(node: str, index: int, occupancy: Occupancy, uuid: str | None = None) -> GPUResource:
    return GPUResource(
        uuid=uuid or f"GPU-{node}-{index}",
        node=node,
        index=index,
        name="A100",
        model="A100",
        memory_gb=40,
        occupancy=occupancy,
        compute_processes=(
            (
                ProcessInfo(
                    pid=100 + index,
                    name="train",
                    gpu_uuid=uuid or f"GPU-{node}-{index}",
                    memory_used_gb=12,
                    username="gkoutr",
                ),
            )
            if occupancy == Occupancy.RESEARCHER
            else ()
        ),
    )


def test_quiet_node_is_zero():
    rows = compute_annoyance([], [_gpu("atlas", 0, Occupancy.AVAILABLE), _gpu("atlas", 1, Occupancy.AVAILABLE)])
    assert rows == [{"node": "atlas", "score": 0, "reasons": []}]


def test_login_foreign_and_live_researcher():
    intents = [
        IntentSignal(kind=IntentKind.LOGIN, node="atlas", username="gkoutr"),
        IntentSignal(kind=IntentKind.FOREIGN_PROCESS, node="atlas", gpu_uuid="GPU-atlas-0", username="gkoutr"),
    ]
    gpus = [_gpu("atlas", 0, Occupancy.RESEARCHER), _gpu("atlas", 1, Occupancy.AVAILABLE)]
    rows = compute_annoyance(intents, gpus)
    # 12*1 + 20*1 + 50*(1/2) = 57
    assert len(rows) == 1
    assert rows[0]["node"] == "atlas"
    assert rows[0]["score"] == 57
    assert rows[0]["reasons"] == [
        "1 LOGIN (gkoutr)",
        "1 FOREIGN_PROCESS",
        "1/2 GPUs RESEARCHER now",
    ]


def test_occupancy_time_when_cards_are_idle_now():
    gpus = [_gpu("uranus", 0, Occupancy.AVAILABLE), _gpu("uranus", 1, Occupancy.AVAILABLE)]
    rows = compute_annoyance(
        [],
        gpus,
        occupancy_seconds={"uranus": {"researcher": 100.0, "total": 400.0}},
    )
    # live_frac=0, time_frac=0.25, raw=12.5 → 13
    assert rows[0]["score"] == 13
    assert rows[0]["reasons"] == ["researcher occupancy 0.25 (100s / 400s)"]


def test_live_occupancy_outranks_low_historical_frac():
    gpus = [_gpu("atlas", 0, Occupancy.RESEARCHER), _gpu("atlas", 1, Occupancy.RESEARCHER)]
    rows = compute_annoyance(
        [],
        gpus,
        occupancy_seconds={"atlas": {"researcher": 10.0, "total": 400.0}},
    )
    # max(1.0, 0.025) → 50
    assert rows[0]["score"] == 50
    assert "2/2 GPUs RESEARCHER now" in rows[0]["reasons"]
    assert "researcher occupancy 0.03 (10s / 400s)" in rows[0]["reasons"]


def test_score_is_deterministic_and_sorted():
    intents = [
        IntentSignal(kind=IntentKind.LOGIN, node="uranus", username="csp"),
        IntentSignal(kind=IntentKind.LOGIN, node="atlas", username="gkoutr"),
        IntentSignal(kind=IntentKind.FOREIGN_PROCESS, node="uranus"),
    ]
    gpus = [_gpu("uranus", 0, Occupancy.AVAILABLE), _gpu("atlas", 0, Occupancy.AVAILABLE)]
    first = compute_annoyance(intents, gpus)
    second = compute_annoyance(list(reversed(intents)), list(reversed(gpus)))
    assert first == second
    assert [row["node"] for row in first] == ["atlas", "uranus"]
    assert first[0]["score"] == 12
    assert first[1]["score"] == 32


def test_duplicate_intents_do_not_inflate_score():
    intents = [
        IntentSignal(kind=IntentKind.LOGIN, node="atlas", username="gkoutr"),
        IntentSignal(kind=IntentKind.LOGIN, node="atlas", username="gkoutr"),
        IntentSignal(kind=IntentKind.FOREIGN_PROCESS, node="atlas", gpu_uuid="GPU-atlas-0", username="gkoutr"),
        IntentSignal(kind=IntentKind.FOREIGN_PROCESS, node="atlas", gpu_uuid="GPU-atlas-0", username="gkoutr"),
    ]
    gpus = [_gpu("atlas", 0, Occupancy.RESEARCHER), _gpu("atlas", 1, Occupancy.AVAILABLE)]
    rows = compute_annoyance(intents, gpus)
    assert rows[0]["score"] == 57


def test_score_caps_at_100():
    intents = [
        IntentSignal(kind=IntentKind.FOREIGN_PROCESS, node="uranus", gpu_uuid=f"g{i}")
        for i in range(8)
    ]
    gpus = [_gpu("uranus", i, Occupancy.RESEARCHER) for i in range(4)]
    rows = compute_annoyance(intents, gpus)
    assert rows[0]["score"] == 100


def test_metrics_accumulate_per_node_researcher_seconds():
    metrics = InfraMetrics()
    metrics.observe_gpus(
        [_gpu("atlas", 0, Occupancy.RESEARCHER), _gpu("atlas", 1, Occupancy.AVAILABLE)],
        10.0,
    )
    metrics.observe_gpus([_gpu("uranus", 0, Occupancy.RESEARCHER)], 5.0)
    occ = metrics.occupancy_seconds()
    assert occ["atlas"] == {"researcher": 10.0, "total": 20.0}
    assert occ["uranus"] == {"researcher": 5.0, "total": 5.0}
    users = {(row["node"], row["username"]): row for row in metrics.user_occupancy()}
    assert users[("atlas", "gkoutr")]["other_s"] == 10.0
    assert users[("uranus", "gkoutr")]["other_s"] == 5.0
    assert users[("atlas", "gkoutr")]["troiani_s"] == 0.0


def test_metrics_survive_control_restart(service):
    service.metrics.observe_gpus([_gpu("atlas", 0, Occupancy.RESEARCHER)], 30.0)
    service.store.set_kv("infra_metrics", service.metrics.to_dict())
    restarted = PlatformService(service.config, store=service.store)
    users = {(row["node"], row["username"]): row for row in restarted.metrics.user_occupancy()}
    assert users[("atlas", "gkoutr")]["seconds"] == 30.0
    assert restarted.metrics.researcher_gpu_seconds == 30.0


def test_status_exposes_annoyance(service):
    service.store.put_gpu(_gpu("atlas", 0, Occupancy.RESEARCHER))
    service.store.put_gpu(_gpu("atlas", 1, Occupancy.AVAILABLE))
    service.store.put_intent(IntentSignal(kind=IntentKind.LOGIN, node="atlas", username="gkoutr"))
    service.store.put_intent(
        IntentSignal(kind=IntentKind.FOREIGN_PROCESS, node="atlas", gpu_uuid="GPU-atlas-0", username="gkoutr")
    )
    client = TestClient(create_app(service.config, service))
    status = client.get("/v1/status").json()
    assert "annoyance" in status
    row = next(item for item in status["annoyance"] if item["node"] == "atlas")
    assert row["score"] == 57
    assert "1 LOGIN (gkoutr)" in row["reasons"]
    assert "1 FOREIGN_PROCESS" in row["reasons"]
    service.metrics.observe_gpus([_gpu("atlas", 0, Occupancy.RESEARCHER)], 12.0)
    status = client.get("/v1/status").json()
    users = {(row["node"], row["username"]): row for row in status["user_occupancy"]}
    assert users[("atlas", "gkoutr")]["seconds"] == 12.0


def test_dashboard_keeps_policy_and_shows_sharing(service):
    client = TestClient(create_app(service.config, service))
    page = client.get("/")
    assert page.status_code == 200
    assert "Sharing" in page.text
    assert "share-body" in page.text
    assert "Policy" in page.text
    assert "Tokens" in page.text
    js = client.get("/static/app.js")
    assert js.status_code == 200
    assert "renderSharing" in js.text
    assert "status.annoyance" in js.text
    assert "Drain" in js.text
    assert "tokens_ingested" in js.text
    assert "user-occ-body" in page.text
    assert "renderUserOccupancy" in js.text
    assert "user_occupancy" in js.text
    assert "theme-toggle" in page.text
    assert "applyTheme" in js.text
    assert "data-theme" in client.get("/static/app.css").text
