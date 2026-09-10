from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from troiani_platform.config import PolicyConfig
from troiani_platform.models import ACTIVE_STATES, Admission, Job, Priority
from troiani_platform.models import GPUResource
from troiani_platform.policy.windows import in_window


class DefaultPolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def allows_admission(
        self,
        job: Job,
        running: Sequence[Job],
        gpus: Sequence[GPUResource],
        now: datetime | None = None,
    ) -> Admission:
        current = now or datetime.now(timezone.utc)
        if job.spec.priority == Priority.RESEARCHER:
            return Admission(False, "researcher priority is an inventory signal, not a Troiani job")
        if job.spec.gpus > self.config.max_gpus_per_job:
            return Admission(False, f"job asks for {job.spec.gpus} GPUs > max_gpus_per_job")
        if getattr(self.config, "stop_all", False):
            return Admission(False, "STOP ALL latched")
        if not in_window(self.config, current):
            return Admission(False, "outside scheduling window")
        active = [item for item in running if item.state in ACTIVE_STATES]
        if len(active) >= self.config.max_jobs:
            return Admission(False, "max simultaneous Troiani jobs reached")
        used = sum(len(item.assigned_gpus) or item.spec.gpus for item in active)
        if used + job.spec.gpus > self.config.max_troiani_gpus:
            return Admission(False, "max Troiani GPU allocation reached")
        return Admission(True, "admitted")

    def runtime_exceeded(self, job: Job, now: datetime | None = None) -> bool:
        limit = job.spec.max_runtime or self.config.max_runtime_s
        if not limit:
            return False
        current = now or datetime.now(timezone.utc)
        return job.runtime_s(current.isoformat()) >= limit
