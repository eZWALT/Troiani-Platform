from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from troiani_platform.models import Job, QUEUED_STATES
from troiani_platform.policy.priorities import rank_jobs


def queued_jobs(jobs: Sequence[Job], now: datetime | None = None) -> list[Job]:
    current = now or datetime.now(timezone.utc)
    ready = []
    for job in jobs:
        if job.state not in QUEUED_STATES:
            continue
        if job.cooldown_until:
            until = datetime.fromisoformat(job.cooldown_until)
            if until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)
            if until > current:
                continue
        ready.append(job)
    return rank_jobs(ready)
