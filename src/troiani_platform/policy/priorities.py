from __future__ import annotations

from typing import Sequence

from troiani_platform.models import Job, Priority


def rank_jobs(jobs: Sequence[Job]) -> list[Job]:
    def key(job: Job) -> tuple:
        prio = 0 if job.spec.priority == Priority.RESEARCHER else 1
        return (prio, job.created_at, job.id)

    return sorted(jobs, key=key)
