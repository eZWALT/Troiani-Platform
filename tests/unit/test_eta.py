from troiani_platform.models import TOKENS_100B, Job, Run, eta_to_100b, format_eta
from tests.conftest import make_spec


def test_eta_to_100b_none_without_rate():
    assert eta_to_100b(1000, None) == (None, None)
    assert eta_to_100b(1000, 0) == (None, None)
    assert eta_to_100b(1000, -1) == (None, None)


def test_eta_to_100b_reached():
    seconds, label = eta_to_100b(TOKENS_100B, 1000)
    assert seconds == 0.0
    assert label == "reached"


def test_eta_to_100b_duration():
    seconds, label = eta_to_100b(0, 1_000_000)
    assert seconds == TOKENS_100B / 1_000_000
    assert label is not None
    assert "d" in label or "y" in label or "h" in label
    assert format_eta(90) == "1m 30s"


def test_job_and_run_expose_eta():
    job = Job(id="job-e", spec=make_spec(name="e"), tokens_ingested=0, tokens_per_sec=2000)
    payload = job.to_dict()
    assert payload["eta_100b"]
    assert payload["eta_100b_s"] == TOKENS_100B / 2000
    run = Run(id="run-e", job_id="job-e", experiment="e", tokens_ingested=TOKENS_100B, tokens_per_sec=10)
    assert run.to_dict()["eta_100b"] == "reached"
