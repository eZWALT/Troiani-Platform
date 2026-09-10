from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from troiani_platform.models import GPUResource, Job, Placement


def place_job(
    job: Job,
    gpus: Sequence[GPUResource],
    reserved: set[str] | None = None,
    cooldown: set[str] | None = None,
    memory_busy_gb: float = 2.0,
) -> Placement | None:
    blocked = set(reserved or ()) | set(cooldown or ())
    candidates = [
        gpu
        for gpu in gpus
        if gpu.uuid not in blocked
        and gpu.safe_to_allocate(memory_busy_gb)
        and gpu.memory_fits(job.spec.min_gpu_memory_gb)
        and gpu.matches_preferred_type(job.spec.preferred_gpu_type)
    ]
    by_node: dict[str, list[GPUResource]] = defaultdict(list)
    for gpu in candidates:
        if job.spec.preferred_node and gpu.node != job.spec.preferred_node:
            continue
        by_node[gpu.node].append(gpu)

    def score(group: list[GPUResource]) -> tuple:
        preferred = sum(1 for gpu in group if job.spec.preferred_gpu_type and gpu.matches_preferred_type(job.spec.preferred_gpu_type))
        free_mem = sum(gpu.memory_gb - gpu.memory_used_gb for gpu in group)
        return (preferred, -abs(len(group) - job.spec.gpus), free_mem)

    for node, group in sorted(by_node.items(), key=lambda item: score(item[1]), reverse=True):
        group_sorted = sorted(group, key=lambda gpu: (gpu.memory_gb, -gpu.utilization, gpu.uuid))
        if len(group_sorted) < job.spec.gpus:
            continue
        chosen = group_sorted[: job.spec.gpus]
        if any(not gpu.memory_fits(job.spec.min_gpu_memory_gb) for gpu in chosen):
            continue
        return Placement(node=node, gpu_uuids=[gpu.uuid for gpu in chosen])
    return None
