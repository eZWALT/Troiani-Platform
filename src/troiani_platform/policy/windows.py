from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from troiani_platform.config import PolicyConfig, TimeWindow

_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _parse_hhmm(value: str) -> time:
    hour, minute = value.split(":", 1)
    return time(int(hour), int(minute))


def _weekday(now: datetime) -> str:
    return _WEEKDAYS[now.weekday()]


def window_contains(window: TimeWindow, now: datetime) -> bool:
    if not window.enabled:
        return False
    current = time(now.hour, now.minute, now.second)
    start = _parse_hhmm(window.start)
    end = _parse_hhmm(window.end)
    weekday = _weekday(now)
    if start <= end:
        if window.days and weekday not in window.days:
            return False
        return start <= current <= end
    # Overnight windows belong to the start weekday (Fri 20:00 → Sat 08:00).
    if window.days:
        previous = _WEEKDAYS[(now.weekday() - 1) % 7]
        if current >= start:
            return weekday in window.days
        return previous in window.days and current <= end
    return current >= start or current <= end


AGGRESSIVENESS = {
    "low": {"memory_busy_gb": 4.0, "cooldown_s": 60, "max_troiani_gpus": 2, "max_jobs": 2},
    "mid": {"memory_busy_gb": 2.0, "cooldown_s": 30, "max_troiani_gpus": 4, "max_jobs": 4},
    "extreme": {"memory_busy_gb": 0.5, "cooldown_s": 5, "max_troiani_gpus": 6, "max_jobs": 6},
}


def default_presets() -> list[TimeWindow]:
    weekdays = ("mon", "tue", "wed", "thu", "fri")
    weekend = ("sat", "sun")
    return [
        TimeWindow("20:00", "08:00", name="weeknights", days=weekdays, enabled=False),
        TimeWindow("12:00", "14:00", name="lunch", days=weekdays, enabled=False),
        TimeWindow("00:00", "23:59", name="weekends", days=weekend, enabled=False),
        TimeWindow("20:00", "08:00", name="weekend-nights", days=weekend, enabled=False),
        TimeWindow("09:00", "11:00", name="special", days=(), enabled=False),
        TimeWindow("00:00", "23:59", name="festive", days=(), enabled=False),
    ]


def rule_preset(name: str) -> TimeWindow | None:
    for window in default_presets():
        if window.name == name:
            return window
    return None


def apply_aggressiveness(policy: PolicyConfig, level: str) -> PolicyConfig:
    key = str(level or "mid").strip().lower()
    if key not in AGGRESSIVENESS:
        key = "mid"
    profile = AGGRESSIVENESS[key]
    policy.aggressiveness = key
    policy.memory_busy_gb = float(profile["memory_busy_gb"])
    policy.cooldown_s = int(profile["cooldown_s"])
    policy.max_troiani_gpus = int(profile["max_troiani_gpus"])
    policy.max_jobs = int(profile["max_jobs"])
    return policy


def merge_presets(windows: list[TimeWindow]) -> list[TimeWindow]:
    have = {window.name for window in windows if window.name}
    merged = list(windows)
    for preset in default_presets():
        if preset.name not in have:
            merged.append(preset)
    return merged


def in_window(policy: PolicyConfig, now: datetime) -> bool:
    if getattr(policy, "stop_all", False):
        return False
    enabled = [window for window in policy.windows if window.enabled]
    if not enabled:
        return True
    tz = ZoneInfo(policy.timezone) if policy.timezone else None
    local = now.astimezone(tz) if tz and now.tzinfo else now
    return any(window_contains(window, local) for window in enabled)
