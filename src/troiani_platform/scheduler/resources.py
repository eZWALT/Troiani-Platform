from __future__ import annotations

from typing import Sequence

from troiani_platform.models import GPUResource, Occupancy


def researcher_conflict_uuids(gpus: Sequence[GPUResource], assigned: set[str]) -> set[str]:
    conflicts: set[str] = set()
    for gpu in gpus:
        if gpu.uuid not in assigned:
            continue
        if gpu.occupancy == Occupancy.RESEARCHER or gpu.researcher_present:
            conflicts.add(gpu.uuid)
    return conflicts


def index_by_uuid(gpus: Sequence[GPUResource]) -> dict[str, GPUResource]:
    return {gpu.uuid: gpu for gpu in gpus}
