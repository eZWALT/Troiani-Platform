from troiani_platform.infra.notation import checkpoint_id, experiment_slug, gpu_ref
from troiani_platform.models import GPUResource, Occupancy


def test_gpu_ref_and_checkpoint_id():
    assert gpu_ref("Atlas", 1) == "atlas/gpu1"
    assert experiment_slug("Exp X / 1") == "exp-x-1"
    assert checkpoint_id("smoke-1gpu", 50, "regular") == "smoke-1gpu-ckpt-000050"
    assert checkpoint_id("smoke-1gpu", 48, "preemption") == "smoke-1gpu-preempt-000048"


def test_gpu_to_dict_exposes_sm_and_vram():
    gpu = GPUResource(
        uuid="GPU-aaa",
        node="atlas",
        index=0,
        name="NVIDIA A100-PCIE-40GB",
        model="A100",
        memory_gb=40,
        utilization=12.0,
        memory_used_gb=26.0,
        memory_util=4.0,
        occupancy=Occupancy.RESEARCHER,
    )
    data = gpu.to_dict()
    assert data["ref"] == "atlas/gpu0"
    assert data["sm"] == 12.0
    assert data["sm_util"] == 12.0
    assert data["vram_gb"] == 40
    assert data["vram_used_gb"] == 26.0
    assert data["vram_pct"] == 65.0
    assert data["mem_ctrl"] == 4.0
    restored = GPUResource.from_dict({"uuid": "GPU-aaa", "node": "atlas", "index": 0, "name": "A100", "model": "A100", "sm": 9, "vram_gb": 80, "vram_used_gb": 8})
    assert restored.utilization == 9
    assert restored.memory_gb == 80
    assert restored.memory_used_gb == 8
