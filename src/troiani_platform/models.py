from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from troiani_platform.errors import JobSpecError
from troiani_platform.infra.notation import gpu_ref


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def elapsed_seconds(started: str | None, ended: str | None = None, now: str | None = None) -> float:
    start = parse_ts(started)
    if start is None:
        return 0.0
    stop = parse_ts(ended) or parse_ts(now) or datetime.now(timezone.utc)
    return max(0.0, (stop - start).total_seconds())


def format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


TOKENS_100B = 100_000_000_000


def format_eta(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days >= 365:
        return f"{total / (365 * 86400):.1f}y"
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def eta_to_100b(ingested: float | None, tokens_per_sec: float | None) -> tuple[float | None, str | None]:
    """Wall-estimate at the current tok/s. None means no rate; 'reached' if past 100B."""
    try:
        rate = float(tokens_per_sec) if tokens_per_sec is not None else 0.0
    except (TypeError, ValueError):
        return None, None
    if rate <= 0:
        return None, None
    have = float(ingested or 0.0)
    if have >= TOKENS_100B:
        return 0.0, "reached"
    seconds = (TOKENS_100B - have) / rate
    return seconds, format_eta(seconds)


def _eta_100b_fields(ingested: float | None, tokens_per_sec: float | None) -> dict[str, Any]:
    seconds, label = eta_to_100b(ingested, tokens_per_sec)
    return {"eta_100b_s": seconds, "eta_100b": label}


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class JobState(str, Enum):
    PENDING = "PENDING"
    SCHEDULED = "SCHEDULED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    CHECKPOINTING = "CHECKPOINTING"
    PREEMPTED = "PREEMPTED"
    QUEUED = "QUEUED"
    RESUMING = "RESUMING"
    PAUSED = "PAUSED"
    ANOMALY = "ANOMALY"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = frozenset({JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED})

ACTIVE_STATES = frozenset(
    {
        JobState.SCHEDULED,
        JobState.STARTING,
        JobState.RUNNING,
        JobState.CHECKPOINTING,
        JobState.RESUMING,
    }
)

QUEUED_STATES = frozenset({JobState.PENDING, JobState.QUEUED, JobState.PREEMPTED, JobState.PAUSED})


class Priority(str, Enum):
    RESEARCHER = "researcher"
    OPPORTUNISTIC = "opportunistic"


class Occupancy(str, Enum):
    AVAILABLE = "AVAILABLE"
    TROIANI = "TROIANI"
    RESEARCHER = "RESEARCHER"
    DRAINING = "DRAINING"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


class CheckpointKind(str, Enum):
    REGULAR = "regular"
    MILESTONE = "milestone"
    PREEMPTION = "preemption"
    BEST = "best"
    HEALTHY = "healthy"
    SHUTDOWN = "shutdown"
    FAILURE = "failure"
    EVAL = "eval"


class CheckpointState(str, Enum):
    WRITING = "WRITING"
    VALIDATING = "VALIDATING"
    VALID = "VALID"
    CORRUPTED = "CORRUPTED"
    SUPERSEDED = "SUPERSEDED"
    EXPIRED = "EXPIRED"


class AnomalyKind(str, Enum):
    LOSS_NAN = "LOSS_NAN"
    LOSS_INF = "LOSS_INF"
    LOSS_SPIKE = "LOSS_SPIKE"
    LOSS_EXPLODING = "LOSS_EXPLODING"
    TRAINING_STALLED = "TRAINING_STALLED"
    THROUGHPUT_COLLAPSE = "THROUGHPUT_COLLAPSE"
    CHECKPOINT_FAILURE = "CHECKPOINT_FAILURE"
    GPU_UNDERUTILIZATION = "GPU_UNDERUTILIZATION"
    DATA_PIPELINE_STARVATION = "DATA_PIPELINE_STARVATION"


class IntentKind(str, Enum):
    LOGIN = "LOGIN"
    FOREIGN_PROCESS = "FOREIGN_PROCESS"
    MEMORY_PRESSURE = "MEMORY_PRESSURE"
    UTIL_SPIKE = "UTIL_SPIKE"
    INTERACTIVE_SESSION = "INTERACTIVE_SESSION"
    RESEARCHER_CLAIM = "RESEARCHER_CLAIM"


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    name: str
    gpu_uuid: str
    memory_used_gb: float
    username: str | None = None
    troiani_owned: bool = False
    command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ProcessInfo":
        return cls(
            pid=int(raw["pid"]),
            name=str(raw.get("name") or ""),
            gpu_uuid=str(raw.get("gpu_uuid") or ""),
            memory_used_gb=float(raw.get("memory_used_gb") or 0),
            username=raw.get("username"),
            troiani_owned=bool(raw.get("troiani_owned")),
            command=raw.get("command"),
        )


@dataclass(frozen=True)
class GPUResource:
    uuid: str
    node: str
    index: int
    name: str
    model: str
    memory_gb: float
    utilization: float = 0.0  # SM / compute busy % (nvidia-smi gpu_util)
    memory_used_gb: float = 0.0
    memory_util: float = 0.0  # memory-controller busy % if NVML reports it
    temperature: float | None = None
    power_w: float | None = None
    compute_processes: tuple[ProcessInfo, ...] = ()
    occupancy: Occupancy = Occupancy.UNKNOWN
    job_id: str | None = None
    worker_id: str | None = None
    updated_at: str = field(default_factory=utcnow)

    @property
    def sm_util(self) -> float:
        return float(self.utilization)

    @property
    def vram_pct(self) -> float:
        return 100.0 * float(self.memory_used_gb) / max(float(self.memory_gb), 1e-6)

    @property
    def researcher_present(self) -> bool:
        return any(not p.troiani_owned for p in self.compute_processes)

    @property
    def troiani_present(self) -> bool:
        return any(p.troiani_owned for p in self.compute_processes)

    def memory_fits(self, min_gpu_memory_gb: float) -> bool:
        return self.memory_gb + 1e-6 >= float(min_gpu_memory_gb)

    def matches_preferred_type(self, preferred: str | None) -> bool:
        if not preferred:
            return True
        token = preferred.strip().lower()
        hay = f"{self.model} {self.name} {int(self.memory_gb)}gb".lower()
        return token in hay

    def safe_to_allocate(self, memory_busy_gb: float = 2.0) -> bool:
        if self.occupancy in {Occupancy.RESEARCHER, Occupancy.UNHEALTHY, Occupancy.DRAINING}:
            return False
        if self.researcher_present:
            return False
        if self.occupancy == Occupancy.TROIANI and self.job_id:
            return False
        if self.compute_processes and self.memory_used_gb >= memory_busy_gb:
            if not all(p.troiani_owned for p in self.compute_processes):
                return False
        if not self.compute_processes and self.memory_used_gb >= memory_busy_gb:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["occupancy"] = self.occupancy.value
        data["compute_processes"] = [p.to_dict() for p in self.compute_processes]
        data["ref"] = gpu_ref(self.node, self.index, self.uuid)
        data["sm"] = self.sm_util
        data["sm_util"] = self.sm_util
        data["vram_gb"] = self.memory_gb
        data["vram_used_gb"] = self.memory_used_gb
        data["vram_pct"] = round(self.vram_pct, 1)
        data["mem_ctrl"] = self.memory_util
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GPUResource":
        procs = tuple(ProcessInfo.from_dict(p) for p in (raw.get("compute_processes") or ()))
        occ = raw.get("occupancy") or Occupancy.UNKNOWN
        memory_gb = float(raw.get("vram_gb") if raw.get("vram_gb") is not None else raw.get("memory_gb") or 0)
        memory_used = float(
            raw.get("vram_used_gb") if raw.get("vram_used_gb") is not None else raw.get("memory_used_gb") or 0
        )
        sm = float(raw.get("sm_util") if raw.get("sm_util") is not None else raw.get("sm") if raw.get("sm") is not None else raw.get("utilization") or 0)
        return cls(
            uuid=str(raw["uuid"]),
            node=str(raw.get("node") or "unknown"),
            index=int(raw.get("index") or 0),
            name=str(raw.get("name") or ""),
            model=str(raw.get("model") or ""),
            memory_gb=memory_gb,
            utilization=sm,
            memory_used_gb=memory_used,
            memory_util=float(raw.get("mem_ctrl") if raw.get("mem_ctrl") is not None else raw.get("memory_util") or 0),
            temperature=raw.get("temperature"),
            power_w=raw.get("power_w"),
            compute_processes=procs,
            occupancy=occ if isinstance(occ, Occupancy) else Occupancy(occ),
            job_id=raw.get("job_id"),
            worker_id=raw.get("worker_id"),
            updated_at=str(raw.get("updated_at") or utcnow()),
        )


@dataclass(frozen=True)
class DatasetFingerprint:
    name: str
    version: str = "0"
    revision: str = ""
    dataset_hash: str = ""
    shards: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "DatasetFingerprint | None":
        if not raw:
            return None
        return cls(
            name=str(raw.get("name") or "unknown"),
            version=str(raw.get("version") or "0"),
            revision=str(raw.get("revision") or ""),
            dataset_hash=str(raw.get("dataset_hash") or raw.get("hash") or ""),
            shards=tuple(raw.get("shards") or ()),
        )


@dataclass(frozen=True)
class EvaluationHook:
    every_steps: int
    command: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"every_steps": self.every_steps, "command": list(self.command)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "EvaluationHook | None":
        if not raw:
            return None
        command = raw.get("command")
        if isinstance(command, str):
            raise JobSpecError("evaluation.command must be a list, not a shell string")
        if not command:
            raise JobSpecError("evaluation.command must be a non-empty argv list")
        return cls(every_steps=int(raw["every_steps"]), command=tuple(str(x) for x in command))


@dataclass(frozen=True)
class JobSpec:
    name: str
    command: tuple[str, ...]
    gpus: int = 1
    min_gpu_memory_gb: float = 0
    preferred_gpu_type: str | None = None
    preferred_node: str = ""
    priority: Priority = Priority.OPPORTUNISTIC
    preemptible: bool = True
    checkpoint_interval: int = 600
    checkpoint_every_steps: int | None = None
    max_runtime: int = 0
    dataset: DatasetFingerprint | None = None
    evaluation: EvaluationHook | None = None
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    experiment: str | None = None

    def validate(self) -> None:
        if not _NAME_RE.match(self.name):
            raise JobSpecError(f"invalid job name: {self.name!r}")
        if not self.command:
            raise JobSpecError("command must be a non-empty argv list")
        if any(not str(part) for part in self.command):
            raise JobSpecError("command contains empty parts")
        if self.gpus < 1:
            raise JobSpecError("gpus must be >= 1")
        if self.min_gpu_memory_gb < 0:
            raise JobSpecError("min_gpu_memory_gb must be >= 0")
        if self.checkpoint_interval < 0:
            raise JobSpecError("checkpoint_interval must be >= 0")
        if self.evaluation is not None:
            if self.evaluation.every_steps < 1:
                raise JobSpecError("evaluation.every_steps must be >= 1")
            if not self.evaluation.command or any(not str(part) for part in self.evaluation.command):
                raise JobSpecError("evaluation.command must be a non-empty argv list")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": list(self.command),
            "gpus": self.gpus,
            "min_gpu_memory_gb": self.min_gpu_memory_gb,
            "preferred_gpu_type": self.preferred_gpu_type,
            "preferred_node": self.preferred_node,
            "priority": self.priority.value,
            "preemptible": self.preemptible,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "max_runtime": self.max_runtime,
            "dataset": self.dataset.to_dict() if self.dataset else None,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "env": dict(self.env),
            "cwd": self.cwd,
            "experiment": self.experiment,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "JobSpec":
        command = raw.get("command")
        if isinstance(command, str):
            raise JobSpecError("command must be a YAML list, not a shell string")
        if not command:
            raise JobSpecError("command is required")
        priority = raw.get("priority") or Priority.OPPORTUNISTIC
        spec = cls(
            name=str(raw.get("name") or "unnamed"),
            command=tuple(str(x) for x in command),
            gpus=int(raw.get("gpus") or 1),
            min_gpu_memory_gb=float(raw.get("min_gpu_memory_gb") or 0),
            preferred_gpu_type=raw.get("preferred_gpu_type"),
            preferred_node=str(raw.get("preferred_node") or ""),
            priority=priority if isinstance(priority, Priority) else Priority(str(priority).lower()),
            preemptible=bool(raw.get("preemptible", True)),
            checkpoint_interval=int(raw.get("checkpoint_interval") or 600),
            checkpoint_every_steps=raw.get("checkpoint_every_steps"),
            max_runtime=int(raw.get("max_runtime") or 0),
            dataset=DatasetFingerprint.from_dict(raw.get("dataset")),
            evaluation=EvaluationHook.from_dict(raw.get("evaluation")),
            env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
            cwd=raw.get("cwd"),
            experiment=raw.get("experiment"),
        )
        spec.validate()
        return spec


@dataclass
class Job:
    id: str
    spec: JobSpec
    state: JobState = JobState.PENDING
    run_id: str | None = None
    assigned_gpus: list[str] = field(default_factory=list)
    assigned_node: str | None = None
    worker_id: str | None = None
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)
    started_at: str | None = None
    ended_at: str | None = None
    cooldown_until: str | None = None
    preempt_requested: bool = False
    last_checkpoint_id: str | None = None
    last_step: int = 0
    last_loss: float | None = None
    tokens_ingested: int = 0
    tokens_per_sec: float | None = None
    rollback_count: int = 0
    message: str = ""
    pid: int | None = None
    trained_s: float = 0.0
    active_since: str | None = None

    def runtime_s(self, now: str | None = None) -> float:
        extra = 0.0
        if self.active_since and self.state in ACTIVE_STATES:
            extra = elapsed_seconds(self.active_since, now=now)
        if self.trained_s or self.active_since:
            return self.trained_s + extra
        return elapsed_seconds(self.started_at, self.ended_at, now=now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "spec": self.spec.to_dict(),
            "state": self.state.value,
            "run_id": self.run_id,
            "assigned_gpus": list(self.assigned_gpus),
            "assigned_node": self.assigned_node,
            "worker_id": self.worker_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "cooldown_until": self.cooldown_until,
            "preempt_requested": self.preempt_requested,
            "last_checkpoint_id": self.last_checkpoint_id,
            "last_step": self.last_step,
            "last_loss": self.last_loss,
            "tokens_ingested": self.tokens_ingested,
            "tokens_per_sec": self.tokens_per_sec,
            **_eta_100b_fields(self.tokens_ingested, self.tokens_per_sec),
            "runtime_s": self.runtime_s(),
            "rollback_count": self.rollback_count,
            "message": self.message,
            "pid": self.pid,
            "trained_s": self.trained_s,
            "active_since": self.active_since,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Job":
        state = raw.get("state") or JobState.PENDING
        return cls(
            id=str(raw["id"]),
            spec=JobSpec.from_dict(raw["spec"]),
            state=state if isinstance(state, JobState) else JobState(state),
            run_id=raw.get("run_id"),
            assigned_gpus=list(raw.get("assigned_gpus") or []),
            assigned_node=raw.get("assigned_node"),
            worker_id=raw.get("worker_id"),
            created_at=str(raw.get("created_at") or utcnow()),
            updated_at=str(raw.get("updated_at") or utcnow()),
            started_at=raw.get("started_at"),
            ended_at=raw.get("ended_at"),
            cooldown_until=raw.get("cooldown_until"),
            preempt_requested=bool(raw.get("preempt_requested")),
            last_checkpoint_id=raw.get("last_checkpoint_id"),
            last_step=int(raw.get("last_step") or 0),
            last_loss=raw.get("last_loss"),
            tokens_ingested=int(raw.get("tokens_ingested") or 0),
            tokens_per_sec=raw.get("tokens_per_sec"),
            rollback_count=int(raw.get("rollback_count") or 0),
            message=str(raw.get("message") or ""),
            pid=raw.get("pid"),
            trained_s=float(raw.get("trained_s") or 0.0),
            active_since=raw.get("active_since"),
        )

    def touch(self, **changes: Any) -> "Job":
        return replace(self, updated_at=utcnow(), **changes)


@dataclass
class Run:
    id: str
    job_id: str
    experiment: str
    status: str = "created"
    git_sha: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)
    started_at: str | None = None
    ended_at: str | None = None
    created_at: str = field(default_factory=utcnow)
    tokens_ingested: int = 0
    tokens_per_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["runtime_s"] = elapsed_seconds(self.started_at, self.ended_at)
        data.update(_eta_100b_fields(self.tokens_ingested, self.tokens_per_sec))
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Run":
        return cls(
            id=str(raw["id"]),
            job_id=str(raw["job_id"]),
            experiment=str(raw.get("experiment") or "default"),
            status=str(raw.get("status") or "created"),
            git_sha=raw.get("git_sha"),
            config=dict(raw.get("config") or {}),
            environment=dict(raw.get("environment") or {}),
            hardware=dict(raw.get("hardware") or {}),
            dataset=raw.get("dataset"),
            metrics=dict(raw.get("metrics") or {}),
            artifacts=list(raw.get("artifacts") or []),
            checkpoints=list(raw.get("checkpoints") or []),
            started_at=raw.get("started_at"),
            ended_at=raw.get("ended_at"),
            created_at=str(raw.get("created_at") or utcnow()),
            tokens_ingested=int(raw.get("tokens_ingested") or 0),
            tokens_per_sec=raw.get("tokens_per_sec"),
        )


@dataclass
class CheckpointRecord:
    id: str
    run_id: str
    job_id: str
    step: int
    epoch: int = 0
    kind: CheckpointKind = CheckpointKind.REGULAR
    state: CheckpointState = CheckpointState.WRITING
    path: str = ""
    healthy: bool = False
    metric: float | None = None
    checksum: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CheckpointRecord":
        kind = raw.get("kind") or CheckpointKind.REGULAR
        state = raw.get("state") or CheckpointState.WRITING
        return cls(
            id=str(raw["id"]),
            run_id=str(raw["run_id"]),
            job_id=str(raw["job_id"]),
            step=int(raw.get("step") or 0),
            epoch=int(raw.get("epoch") or 0),
            kind=kind if isinstance(kind, CheckpointKind) else CheckpointKind(kind),
            state=state if isinstance(state, CheckpointState) else CheckpointState(state),
            path=str(raw.get("path") or ""),
            healthy=bool(raw.get("healthy")),
            metric=raw.get("metric"),
            checksum=str(raw.get("checksum") or ""),
            manifest=dict(raw.get("manifest") or {}),
            created_at=str(raw.get("created_at") or utcnow()),
        )


@dataclass
class Event:
    type: str
    job_id: str | None = None
    run_id: str | None = None
    worker_id: str | None = None
    source: str = "control"
    payload: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=utcnow)
    id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Event":
        return cls(
            type=str(raw["type"]),
            job_id=raw.get("job_id"),
            run_id=raw.get("run_id"),
            worker_id=raw.get("worker_id"),
            source=str(raw.get("source") or "control"),
            payload=dict(raw.get("payload") or {}),
            ts=str(raw.get("ts") or utcnow()),
            id=raw.get("id"),
        )


@dataclass
class Artifact:
    id: str
    run_id: str
    kind: str
    path: str
    checksum: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Artifact":
        return cls(
            id=str(raw["id"]),
            run_id=str(raw["run_id"]),
            kind=str(raw.get("kind") or "file"),
            path=str(raw.get("path") or ""),
            checksum=str(raw.get("checksum") or ""),
            meta=dict(raw.get("meta") or {}),
            created_at=str(raw.get("created_at") or utcnow()),
        )


@dataclass
class WorkerRecord:
    id: str
    node: str
    hostname: str
    last_heartbeat: str = field(default_factory=utcnow)
    gpus: list[str] = field(default_factory=list)
    status: str = "alive"
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "WorkerRecord":
        return cls(
            id=str(raw["id"]),
            node=str(raw.get("node") or ""),
            hostname=str(raw.get("hostname") or ""),
            last_heartbeat=str(raw.get("last_heartbeat") or utcnow()),
            gpus=list(raw.get("gpus") or []),
            status=str(raw.get("status") or "alive"),
            message=str(raw.get("message") or ""),
        )


@dataclass
class Placement:
    node: str
    gpu_uuids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.node, "gpu_uuids": list(self.gpu_uuids)}


@dataclass
class SchedulerDecision:
    kind: str
    job_id: str
    reason: str
    placement: Placement | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "job_id": self.job_id,
            "reason": self.reason,
            "placement": self.placement.to_dict() if self.placement else None,
            "extra": dict(self.extra),
        }


@dataclass
class IntentSignal:
    kind: IntentKind
    node: str
    gpu_uuid: str | None = None
    username: str | None = None
    detail: str = ""
    score: float = 1.0
    ts: str = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "IntentSignal":
        kind = raw.get("kind") or IntentKind.FOREIGN_PROCESS
        return cls(
            kind=kind if isinstance(kind, IntentKind) else IntentKind(kind),
            node=str(raw.get("node") or ""),
            gpu_uuid=raw.get("gpu_uuid"),
            username=raw.get("username"),
            detail=str(raw.get("detail") or ""),
            score=float(raw.get("score") or 1.0),
            ts=str(raw.get("ts") or utcnow()),
        )


@dataclass
class Admission:
    allowed: bool
    reason: str = ""
