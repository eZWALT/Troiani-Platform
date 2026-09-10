from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Watchdog:
    heartbeat_s: float
    stale_s: float
    control_lost_grace_s: float
    last_control_ok: float = 0.0
    last_local_beat: float = 0.0

    def __post_init__(self) -> None:
        now = time.time()
        self.last_control_ok = now
        self.last_local_beat = now

    def beat(self) -> None:
        self.last_local_beat = time.time()

    def mark_control(self, ok: bool) -> None:
        if ok:
            self.last_control_ok = time.time()

    def control_lost(self, now: float | None = None) -> bool:
        current = now if now is not None else time.time()
        return (current - self.last_control_ok) > self.control_lost_grace_s

    def local_stale(self, now: float | None = None) -> bool:
        current = now if now is not None else time.time()
        return (current - self.last_local_beat) > self.stale_s
