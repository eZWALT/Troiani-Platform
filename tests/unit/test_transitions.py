import pytest

from troiani_platform.errors import TransitionError
from troiani_platform.models import Job, JobState
from troiani_platform.state.transitions import apply_transition, validate_transition
from tests.conftest import make_spec


def test_valid_preempt_cycle():
    job = Job(id="j1", spec=make_spec())
    for state in (
        JobState.QUEUED,
        JobState.SCHEDULED,
        JobState.STARTING,
        JobState.RUNNING,
        JobState.CHECKPOINTING,
        JobState.PREEMPTED,
        JobState.QUEUED,
        JobState.SCHEDULED,
        JobState.RESUMING,
        JobState.RUNNING,
        JobState.COMPLETED,
    ):
        job = apply_transition(job, state)
    assert job.state is JobState.COMPLETED


def test_invalid_transition_rejected():
    with pytest.raises(TransitionError):
        validate_transition(JobState.COMPLETED, JobState.RUNNING)


def test_idempotent_same_state():
    job = Job(id="j1", spec=make_spec(), state=JobState.RUNNING)
    again = apply_transition(job, JobState.RUNNING)
    assert again.state is JobState.RUNNING
