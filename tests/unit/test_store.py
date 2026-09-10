from troiani_platform.models import Event, Job
from troiani_platform.state.store import StateStore
from tests.conftest import make_spec


def test_job_and_event_roundtrip(tmp_path):
    store = StateStore(tmp_path / "db.sqlite")
    job = Job(id="job-1", spec=make_spec())
    store.put_job(job)
    assert store.get_job("job-1").spec.name == "job-a"
    store.append_event(Event(type="JOB_SUBMITTED", job_id="job-1"))
    events = store.list_events()
    assert events[0].type == "JOB_SUBMITTED"
    store.close()
