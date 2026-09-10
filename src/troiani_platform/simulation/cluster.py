from __future__ import annotations

from dataclasses import replace

from troiani_platform.models import GPUResource, Occupancy, ProcessInfo, utcnow


def fake_gpu(uuid: str, node: str, index: int, memory_gb: float, model: str = "A100") -> GPUResource:
    return GPUResource(
        uuid=uuid,
        node=node,
        index=index,
        name=f"NVIDIA {model} {int(memory_gb)}GB",
        model=model,
        memory_gb=memory_gb,
        occupancy=Occupancy.AVAILABLE,
    )


def default_pool() -> list[GPUResource]:
    gpus = [
        fake_gpu("gpu-atlas-0", "atlas", 0, 40),
        fake_gpu("gpu-atlas-1", "atlas", 1, 40),
        fake_gpu("gpu-uranus-0", "uranus", 0, 80),
        fake_gpu("gpu-uranus-1", "uranus", 1, 80),
        fake_gpu("gpu-uranus-2", "uranus", 2, 80),
        fake_gpu("gpu-uranus-3", "uranus", 3, 80),
    ]
    return gpus


class SimulatedCluster:
    def __init__(self, gpus: list[GPUResource] | None = None) -> None:
        self.gpus = gpus or default_pool()

    def researcher_claim(self, uuid: str, username: str = "csp") -> GPUResource:
        updated = []
        claimed = None
        for gpu in self.gpus:
            if gpu.uuid != uuid:
                updated.append(gpu)
                continue
            proc = ProcessInfo(
                pid=9999,
                name="vllm",
                gpu_uuid=uuid,
                memory_used_gb=min(gpu.memory_gb - 1, 40),
                username=username,
                troiani_owned=False,
                command="vllm serve",
            )
            gpu = replace(
                gpu,
                occupancy=Occupancy.RESEARCHER,
                compute_processes=(proc,),
                memory_used_gb=proc.memory_used_gb,
                utilization=90.0,
                updated_at=utcnow(),
            )
            claimed = gpu
            updated.append(gpu)
        self.gpus = updated
        return claimed or self.gpus[0]

    def release(self, uuid: str) -> None:
        self.gpus = [
            replace(gpu, occupancy=Occupancy.AVAILABLE, compute_processes=(), memory_used_gb=0, utilization=0)
            if gpu.uuid == uuid
            else gpu
            for gpu in self.gpus
        ]
