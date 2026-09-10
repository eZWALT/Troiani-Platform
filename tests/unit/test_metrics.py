from troiani_platform.models import GPUResource, Occupancy
from troiani_platform.monitoring.metrics import InfraMetrics


def test_observe_util_keeps_ts_sm_vram_and_ref():
    metrics = InfraMetrics()
    gpu = GPUResource(
        uuid="GPU-aaa",
        node="Atlas",
        index=0,
        name="NVIDIA A100-PCIE-40GB",
        model="A100",
        memory_gb=40,
        utilization=12.0,
        memory_used_gb=26.0,
        occupancy=Occupancy.RESEARCHER,
    )
    metrics.observe_util([gpu], "2026-09-10T17:00:00+00:00")
    snap = metrics.snapshot()
    hist = snap["history"]
    assert len(hist) == 1
    sample = hist[0]
    assert sample["ts"] == "2026-09-10T17:00:00+00:00"
    point = sample["Atlas:0"]
    assert point["sm"] == 12.0
    assert point["vram"] == 65.0
    assert point["ref"] == "atlas/gpu0"
    assert "util" in point
    assert point["sm"] != point["vram"]


def test_observe_util_does_not_collapse_sm_and_vram():
    metrics = InfraMetrics()
    idle = GPUResource(
        uuid="GPU-bbb",
        node="uranus",
        index=1,
        name="NVIDIA A100-SXM4-80GB",
        model="A100",
        memory_gb=80,
        utilization=3.0,
        memory_used_gb=72.0,
        occupancy=Occupancy.RESEARCHER,
    )
    busy = GPUResource(
        uuid="GPU-ccc",
        node="uranus",
        index=2,
        name="NVIDIA A100-SXM4-80GB",
        model="A100",
        memory_gb=80,
        utilization=91.0,
        memory_used_gb=70.0,
        occupancy=Occupancy.RESEARCHER,
    )
    metrics.observe_util([idle, busy], "2026-09-10T17:00:01+00:00")
    sample = metrics.snapshot()["history"][0]
    assert sample["uranus:1"]["ref"] == "uranus/gpu1"
    assert sample["uranus:1"]["sm"] == 3.0
    assert sample["uranus:1"]["vram"] == 90.0
    assert sample["uranus:2"]["ref"] == "uranus/gpu2"
    assert sample["uranus:2"]["sm"] == 91.0
