from __future__ import annotations

from typing import Sequence

from troiani_platform.models import CheckpointRecord, CheckpointState


def latest_valid(records: Sequence[CheckpointRecord]) -> CheckpointRecord | None:
    valid = [r for r in records if r.state == CheckpointState.VALID]
    if not valid:
        return None
    return max(valid, key=lambda r: (r.step, r.created_at))


def select_healthy(records: Sequence[CheckpointRecord]) -> CheckpointRecord | None:
    healthy = [r for r in records if r.state == CheckpointState.VALID and r.healthy]
    if not healthy:
        return latest_valid(records)
    return max(healthy, key=lambda r: (r.step, r.created_at))
