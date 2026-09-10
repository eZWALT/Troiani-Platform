from datetime import datetime, timezone

from troiani_platform.config import PolicyConfig, TimeWindow
from troiani_platform.models import Job, JobState
from troiani_platform.policy.policy import DefaultPolicy
from troiani_platform.policy.windows import apply_aggressiveness, default_presets, in_window, merge_presets
from tests.conftest import make_spec


def test_window_overnight():
    policy = PolicyConfig(windows=[TimeWindow("20:00", "08:00")], timezone="UTC")
    night = datetime(2026, 1, 1, 22, 0, tzinfo=timezone.utc)
    noon = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert in_window(policy, night)
    assert not in_window(policy, noon)


def test_named_weekday_overnight_window():
    policy = PolicyConfig(
        windows=[TimeWindow("20:00", "08:00", name="weeknights", days=("mon", "tue", "wed", "thu", "fri"))],
        timezone="UTC",
    )
    friday_night = datetime(2026, 1, 2, 22, 0, tzinfo=timezone.utc)
    saturday_morning = datetime(2026, 1, 3, 7, 0, tzinfo=timezone.utc)
    saturday_noon = datetime(2026, 1, 3, 12, 0, tzinfo=timezone.utc)
    sunday_night = datetime(2026, 1, 4, 22, 0, tzinfo=timezone.utc)
    assert in_window(policy, friday_night)
    assert in_window(policy, saturday_morning)
    assert not in_window(policy, saturday_noon)
    assert not in_window(policy, sunday_night)


def test_all_disabled_windows_mean_always_on():
    policy = PolicyConfig(windows=[TimeWindow("20:00", "08:00", enabled=False)], timezone="UTC")
    noon = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert in_window(policy, noon)


def test_enabled_window_still_closes_outside_hours():
    policy = PolicyConfig(
        windows=[
            TimeWindow("20:00", "08:00", name="weeknights", enabled=True),
            TimeWindow("12:00", "14:00", name="lunch", enabled=False),
        ],
        timezone="UTC",
    )
    noon = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    night = datetime(2026, 1, 1, 22, 0, tzinfo=timezone.utc)
    assert not in_window(policy, noon)
    assert in_window(policy, night)


def test_runtime_limit_uses_useful_time_not_wall_clock():
    policy = DefaultPolicy(PolicyConfig())
    started = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    now = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)
    job = Job(
        id="long-wait",
        spec=make_spec(max_runtime=30),
        state=JobState.RESUMING,
        started_at=started.isoformat(),
        trained_s=5.0,
        active_since=datetime(2026, 1, 1, 0, 9, 50, tzinfo=timezone.utc).isoformat(),
    )
    assert not policy.runtime_exceeded(job, now)
    job.trained_s = 40.0
    assert policy.runtime_exceeded(job, now)


def test_weekend_preset_disabled_by_default():
    names = {w.name: w for w in default_presets()}
    assert "weekends" in names
    assert names["weekends"].days == ("sat", "sun")
    assert names["weekends"].enabled is False
    merged = merge_presets([names["weeknights"]])
    assert {"weeknights", "lunch", "weekends", "weekend-nights", "special", "festive"} <= {w.name for w in merged}


def test_weekend_window_opens_saturday():
    policy = PolicyConfig(
        windows=[TimeWindow("00:00", "23:59", name="weekends", days=("sat", "sun"), enabled=True)],
        timezone="UTC",
    )
    saturday = datetime(2026, 1, 3, 15, 0, tzinfo=timezone.utc)
    monday = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)
    assert in_window(policy, saturday)
    assert not in_window(policy, monday)


def test_stop_all_closes_window_even_if_rules_disabled():
    policy = PolicyConfig(windows=[TimeWindow("20:00", "08:00", enabled=False)], timezone="UTC", stop_all=True)
    noon = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert not in_window(policy, noon)


def test_aggressiveness_maps_existing_knobs():
    policy = PolicyConfig()
    apply_aggressiveness(policy, "extreme")
    assert policy.aggressiveness == "extreme"
    assert policy.memory_busy_gb == 0.5
    assert policy.cooldown_s == 5
    apply_aggressiveness(policy, "low")
    assert policy.max_troiani_gpus == 2


def test_admission_respects_caps():
    policy = DefaultPolicy(PolicyConfig(max_jobs=1, max_troiani_gpus=2, max_gpus_per_job=2))
    running = [Job(id="a", spec=make_spec(gpus=2), state=JobState.RUNNING, assigned_gpus=["u0", "u1"])]
    waiting = Job(id="b", spec=make_spec(name="job-b", gpus=1), state=JobState.PENDING)
    assert not policy.allows_admission(waiting, running, []).allowed
