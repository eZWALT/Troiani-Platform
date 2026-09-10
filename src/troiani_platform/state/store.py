from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from troiani_platform.errors import NotFound
from troiani_platform.models import (
    Artifact,
    CheckpointRecord,
    Event,
    GPUResource,
    IntentSignal,
    Job,
    Run,
    WorkerRecord,
    utcnow,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    state TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    job_id TEXT
);
CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    run_id TEXT,
    state TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    type TEXT NOT NULL,
    job_id TEXT,
    run_id TEXT,
    worker_id TEXT,
    source TEXT,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gpus (
    uuid TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    node TEXT,
    occupancy TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    last_heartbeat TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    run_id TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    job_id TEXT,
    run_id TEXT,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    step INTEGER,
    tags TEXT
);
CREATE TABLE IF NOT EXISTS intents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    payload TEXT NOT NULL,
    node TEXT,
    kind TEXT
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _dump(obj: Any) -> str:
    if hasattr(obj, "to_dict"):
        return json.dumps(obj.to_dict(), default=str)
    return json.dumps(obj, default=str)


class StateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, tuple(params))

    def put_job(self, job: Job) -> Job:
        with self._lock:
            self._execute(
                "INSERT OR REPLACE INTO jobs(id, payload, state, updated_at) VALUES (?, ?, ?, ?)",
                (job.id, _dump(job), job.state.value, job.updated_at),
            )
            self._conn.commit()
        return job

    def get_job(self, job_id: str) -> Job:
        with self._lock:
            row = self._execute("SELECT payload FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFound(f"job {job_id} not found")
        return Job.from_dict(json.loads(row["payload"]))

    def list_jobs(self) -> list[Job]:
        with self._lock:
            rows = self._execute("SELECT payload FROM jobs ORDER BY updated_at DESC").fetchall()
        return [Job.from_dict(json.loads(r["payload"])) for r in rows]

    def put_run(self, run: Run) -> Run:
        with self._lock:
            self._execute(
                "INSERT OR REPLACE INTO runs(id, payload, job_id) VALUES (?, ?, ?)",
                (run.id, _dump(run), run.job_id),
            )
            self._conn.commit()
        return run

    def get_run(self, run_id: str) -> Run:
        with self._lock:
            row = self._execute("SELECT payload FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise NotFound(f"run {run_id} not found")
        return Run.from_dict(json.loads(row["payload"]))

    def list_runs(self) -> list[Run]:
        with self._lock:
            rows = self._execute("SELECT payload FROM runs").fetchall()
        return [Run.from_dict(json.loads(r["payload"])) for r in rows]

    def put_checkpoint(self, record: CheckpointRecord) -> CheckpointRecord:
        with self._lock:
            self._execute(
                "INSERT OR REPLACE INTO checkpoints(id, payload, run_id, state) VALUES (?, ?, ?, ?)",
                (record.id, _dump(record), record.run_id, record.state.value),
            )
            self._conn.commit()
        return record

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord:
        with self._lock:
            row = self._execute("SELECT payload FROM checkpoints WHERE id = ?", (checkpoint_id,)).fetchone()
        if row is None:
            raise NotFound(f"checkpoint {checkpoint_id} not found")
        return CheckpointRecord.from_dict(json.loads(row["payload"]))

    def list_checkpoints(self, run_id: str | None = None) -> list[CheckpointRecord]:
        with self._lock:
            if run_id:
                rows = self._execute(
                    "SELECT payload FROM checkpoints WHERE run_id = ?", (run_id,)
                ).fetchall()
            else:
                rows = self._execute("SELECT payload FROM checkpoints").fetchall()
        return [CheckpointRecord.from_dict(json.loads(r["payload"])) for r in rows]

    def append_event(self, event: Event) -> Event:
        with self._lock:
            cur = self._execute(
                """INSERT INTO events(ts, type, job_id, run_id, worker_id, source, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.ts,
                    event.type,
                    event.job_id,
                    event.run_id,
                    event.worker_id,
                    event.source,
                    json.dumps(event.payload, default=str),
                ),
            )
            self._conn.commit()
            event.id = cur.lastrowid
        return event

    def list_events(self, limit: int = 200, job_id: str | None = None) -> list[Event]:
        with self._lock:
            if job_id:
                rows = self._execute(
                    "SELECT * FROM events WHERE job_id = ? ORDER BY id DESC LIMIT ?",
                    (job_id, limit),
                ).fetchall()
            else:
                rows = self._execute(
                    "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        events = []
        for row in rows:
            events.append(
                Event(
                    id=row["id"],
                    ts=row["ts"],
                    type=row["type"],
                    job_id=row["job_id"],
                    run_id=row["run_id"],
                    worker_id=row["worker_id"],
                    source=row["source"],
                    payload=json.loads(row["payload"] or "{}"),
                )
            )
        return events

    def put_gpu(self, gpu: GPUResource) -> GPUResource:
        with self._lock:
            self._execute(
                "INSERT OR REPLACE INTO gpus(uuid, payload, node, occupancy, updated_at) VALUES (?, ?, ?, ?, ?)",
                (gpu.uuid, _dump(gpu), gpu.node, gpu.occupancy.value, gpu.updated_at),
            )
            self._conn.commit()
        return gpu

    def list_gpus(self) -> list[GPUResource]:
        with self._lock:
            rows = self._execute("SELECT payload FROM gpus").fetchall()
        return [GPUResource.from_dict(json.loads(r["payload"])) for r in rows]

    def put_worker(self, worker: WorkerRecord) -> WorkerRecord:
        with self._lock:
            self._execute(
                "INSERT OR REPLACE INTO workers(id, payload, last_heartbeat) VALUES (?, ?, ?)",
                (worker.id, _dump(worker), worker.last_heartbeat),
            )
            self._conn.commit()
        return worker

    def list_workers(self) -> list[WorkerRecord]:
        with self._lock:
            rows = self._execute("SELECT payload FROM workers").fetchall()
        return [WorkerRecord.from_dict(json.loads(r["payload"])) for r in rows]

    def put_artifact(self, artifact: Artifact) -> Artifact:
        with self._lock:
            self._execute(
                "INSERT OR REPLACE INTO artifacts(id, payload, run_id) VALUES (?, ?, ?)",
                (artifact.id, _dump(artifact), artifact.run_id),
            )
            self._conn.commit()
        return artifact

    def list_artifacts(self, run_id: str | None = None) -> list[Artifact]:
        with self._lock:
            if run_id:
                rows = self._execute(
                    "SELECT payload FROM artifacts WHERE run_id = ?", (run_id,)
                ).fetchall()
            else:
                rows = self._execute("SELECT payload FROM artifacts").fetchall()
        return [Artifact.from_dict(json.loads(r["payload"])) for r in rows]

    def log_metric(
        self,
        name: str,
        value: float,
        job_id: str | None = None,
        run_id: str | None = None,
        step: int | None = None,
        tags: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._execute(
                "INSERT INTO metrics(ts, job_id, run_id, name, value, step, tags) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (utcnow(), job_id, run_id, name, float(value), step, json.dumps(tags or {})),
            )
            self._conn.commit()

    def list_metrics(self, run_id: str | None = None, limit: int = 2000) -> list[dict[str, Any]]:
        with self._lock:
            if run_id:
                rows = self._execute(
                    "SELECT * FROM metrics WHERE run_id = ? ORDER BY id DESC LIMIT ?",
                    (run_id, limit),
                ).fetchall()
            else:
                rows = self._execute(
                    "SELECT * FROM metrics ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        return [dict(r) for r in rows]

    def put_intent(self, signal: IntentSignal) -> IntentSignal:
        with self._lock:
            self._execute(
                "INSERT INTO intents(ts, payload, node, kind) VALUES (?, ?, ?, ?)",
                (signal.ts, _dump(signal), signal.node, signal.kind.value),
            )
            self._conn.commit()
        return signal

    def list_intents(self, limit: int = 100) -> list[IntentSignal]:
        with self._lock:
            rows = self._execute(
                "SELECT payload FROM intents ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [IntentSignal.from_dict(json.loads(r["payload"])) for r in rows]

    def set_kv(self, key: str, value: Any) -> None:
        with self._lock:
            self._execute(
                "INSERT OR REPLACE INTO kv(key, value) VALUES (?, ?)",
                (key, json.dumps(value, default=str)),
            )
            self._conn.commit()

    def get_kv(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row["value"])
