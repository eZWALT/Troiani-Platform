from __future__ import annotations

from typing import Any

from troiani_platform.state.store import StateStore


class Telemetry:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def log(self, name: str, value: float, run_id: str | None = None, job_id: str | None = None, step: int | None = None) -> None:
        self.store.log_metric(name, value, job_id=job_id, run_id=run_id, step=step)

    def latest(self, run_id: str) -> dict[str, Any]:
        rows = self.store.list_metrics(run_id=run_id, limit=200)
        out: dict[str, Any] = {}
        for row in rows:
            out.setdefault(row["name"], row["value"])
        return out
