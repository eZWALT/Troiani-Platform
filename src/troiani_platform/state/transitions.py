from __future__ import annotations

from troiani_platform.errors import TransitionError
from troiani_platform.models import Job, JobState, utcnow

VALID: dict[JobState, frozenset[JobState]] = {
    JobState.PENDING: frozenset(
        {
            JobState.SCHEDULED,
            JobState.QUEUED,
            JobState.CANCELLED,
            JobState.PAUSED,
            JobState.COMPLETED,
            JobState.FAILED,
        }
    ),
    JobState.SCHEDULED: frozenset(
        {
            JobState.STARTING,
            JobState.QUEUED,
            JobState.CANCELLED,
            JobState.PREEMPTED,
            JobState.COMPLETED,
            JobState.FAILED,
        }
    ),
    JobState.STARTING: frozenset(
        {
            JobState.RUNNING,
            JobState.CHECKPOINTING,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.PREEMPTED,
            JobState.COMPLETED,
        }
    ),
    JobState.RUNNING: frozenset(
        {
            JobState.CHECKPOINTING,
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.ANOMALY,
            JobState.PAUSED,
            JobState.PREEMPTED,
        }
    ),
    JobState.CHECKPOINTING: frozenset(
        {
            JobState.RUNNING,
            JobState.PREEMPTED,
            JobState.PAUSED,
            JobState.FAILED,
            JobState.COMPLETED,
            JobState.CANCELLED,
            JobState.ANOMALY,
        }
    ),
    JobState.PREEMPTED: frozenset(
        {JobState.QUEUED, JobState.PENDING, JobState.CANCELLED, JobState.COMPLETED, JobState.FAILED}
    ),
    JobState.QUEUED: frozenset(
        {
            JobState.SCHEDULED,
            JobState.PENDING,
            JobState.CANCELLED,
            JobState.PAUSED,
            JobState.COMPLETED,
            JobState.FAILED,
        }
    ),
    JobState.RESUMING: frozenset(
        {
            JobState.RUNNING,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.CHECKPOINTING,
            JobState.COMPLETED,
        }
    ),
    JobState.PAUSED: frozenset(
        {
            JobState.QUEUED,
            JobState.PENDING,
            JobState.CANCELLED,
            JobState.RESUMING,
            JobState.COMPLETED,
            JobState.FAILED,
        }
    ),
    JobState.ANOMALY: frozenset(
        {
            JobState.PAUSED,
            JobState.CHECKPOINTING,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.QUEUED,
            JobState.COMPLETED,
        }
    ),
    JobState.COMPLETED: frozenset(),
    JobState.FAILED: frozenset({JobState.QUEUED, JobState.PENDING}),
    JobState.CANCELLED: frozenset(),
}

# Recovery / resume path: queued jobs become scheduled, then starting or resuming.
VALID[JobState.SCHEDULED] = VALID[JobState.SCHEDULED] | frozenset({JobState.RESUMING})
VALID[JobState.PREEMPTED] = VALID[JobState.PREEMPTED] | frozenset({JobState.RESUMING})
VALID[JobState.QUEUED] = VALID[JobState.QUEUED] | frozenset({JobState.RESUMING})


def validate_transition(current: JobState, target: JobState) -> None:
    if current == target:
        return
    allowed = VALID.get(current, frozenset())
    if target not in allowed:
        raise TransitionError(f"invalid transition {current.value} -> {target.value}")


def apply_transition(job: Job, target: JobState, message: str = "") -> Job:
    validate_transition(job.state, target)
    now = utcnow()
    started = job.started_at
    ended = job.ended_at
    if target in {JobState.RUNNING, JobState.STARTING, JobState.RESUMING} and not started:
        started = now
    if target in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}:
        ended = now
    if target in {JobState.QUEUED, JobState.PENDING, JobState.PREEMPTED} and job.state != target:
        # Released back to the queue — drop live assignment except last known GPUs for events.
        pass
    return job.touch(state=target, message=message or job.message, started_at=started, ended_at=ended)
