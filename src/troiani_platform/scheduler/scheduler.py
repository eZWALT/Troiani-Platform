from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from troiani_platform.config import PolicyConfig
from troiani_platform.models import ACTIVE_STATES, GPUResource, Job, JobState, Priority, SchedulerDecision
from troiani_platform.policy.policy import DefaultPolicy
from troiani_platform.scheduler.placement import place_job
from troiani_platform.scheduler.queue import queued_jobs
from troiani_platform.scheduler.resources import researcher_conflict_uuids


class CooperativeScheduler:
    """Researcher-first opportunistic scheduler. Never assumes exclusive GPUs."""

    def __init__(self, policy: PolicyConfig | DefaultPolicy) -> None:
        self.policy = policy if isinstance(policy, DefaultPolicy) else DefaultPolicy(policy)

    def schedule(
        self,
        jobs: Sequence[Job],
        gpus: Sequence[GPUResource],
        now: datetime | None = None,
        cooldown_gpus: set[str] | None = None,
    ) -> list[SchedulerDecision]:
        current = now or datetime.now(timezone.utc)
        cooldown = set(cooldown_gpus or ())
        decisions: list[SchedulerDecision] = []
        reserved: set[str] = set()

        for job in jobs:
            if job.state in ACTIVE_STATES:
                reserved.update(job.assigned_gpus)
            if job.cooldown_until:
                until = datetime.fromisoformat(job.cooldown_until)
                if until.tzinfo is None:
                    until = until.replace(tzinfo=timezone.utc)
                if until > current:
                    reserved.update(job.assigned_gpus)

        assigned = {uuid for job in jobs if job.state in ACTIVE_STATES for uuid in job.assigned_gpus}
        conflicts = researcher_conflict_uuids(gpus, assigned)
        if conflicts:
            for job in self._select_preempt_victims(jobs, conflicts):
                decisions.append(
                    SchedulerDecision(
                        kind="PREEMPT",
                        job_id=job.id,
                        reason="researcher present on assigned GPU",
                        extra={"gpus": [uuid for uuid in job.assigned_gpus if uuid in conflicts]},
                    )
                )
                reserved.difference_update(job.assigned_gpus)
                cooldown.update(job.assigned_gpus)

        for job in jobs:
            if job.state in ACTIVE_STATES and self.policy.runtime_exceeded(job, current):
                decisions.append(
                    SchedulerDecision(kind="PREEMPT", job_id=job.id, reason="max runtime exceeded")
                )
                reserved.difference_update(job.assigned_gpus)

        running = [job for job in jobs if job.state in ACTIVE_STATES]
        for job in queued_jobs(jobs, current):
            if job.spec.priority == Priority.RESEARCHER:
                continue
            if job.state == JobState.PAUSED and not job.message.startswith("resume:"):
                continue
            admission = self.policy.allows_admission(job, running, gpus, current)
            if not admission.allowed:
                decisions.append(SchedulerDecision(kind="HOLD", job_id=job.id, reason=admission.reason))
                continue
            placement = place_job(
                job,
                gpus,
                reserved=reserved,
                cooldown=cooldown,
                memory_busy_gb=self.policy.config.memory_busy_gb,
            )
            if placement is None:
                decisions.append(
                    SchedulerDecision(kind="HOLD", job_id=job.id, reason="no matching idle GPUs")
                )
                continue
            decisions.append(
                SchedulerDecision(
                    kind="ALLOCATE",
                    job_id=job.id,
                    reason="idle capacity matches requirements",
                    placement=placement,
                )
            )
            reserved.update(placement.gpu_uuids)
            running.append(job)
        return decisions

    def _select_preempt_victims(self, jobs: Sequence[Job], conflicts: set[str]) -> list[Job]:
        victims = [
            job
            for job in jobs
            if job.state in ACTIVE_STATES
            and job.spec.preemptible
            and set(job.assigned_gpus) & conflicts
        ]
        victims.sort(key=lambda job: (-len(job.assigned_gpus), job.started_at or job.created_at))
        return victims
