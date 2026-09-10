from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class RetentionConfig:
    last_regular: int = 5
    last_milestone: int = 3
    best_n: int = 3
    keep_latest_preemption: bool = True
    keep_latest_healthy: bool = True


_DAY_ALIASES = {
    "mon": "mon",
    "monday": "mon",
    "tue": "tue",
    "tues": "tue",
    "tuesday": "tue",
    "wed": "wed",
    "wednesday": "wed",
    "thu": "thu",
    "thur": "thu",
    "thurs": "thu",
    "thursday": "thu",
    "fri": "fri",
    "friday": "fri",
    "sat": "sat",
    "saturday": "sat",
    "sun": "sun",
    "sunday": "sun",
}


def normalize_weekdays(raw: Any) -> tuple[str, ...]:
    if not raw:
        return ()
    days: list[str] = []
    for item in raw:
        key = str(item).strip().lower()
        if key not in _DAY_ALIASES:
            raise ValueError(f"unknown weekday: {item}")
        alias = _DAY_ALIASES[key]
        if alias not in days:
            days.append(alias)
    return tuple(days)


@dataclass
class TimeWindow:
    start: str
    end: str
    name: str = ""
    days: tuple[str, ...] = ()
    enabled: bool = True

    @classmethod
    def from_mapping(cls, raw: Any) -> "TimeWindow":
        if isinstance(raw, str) and "-" in raw:
            start, end = raw.split("-", 1)
            return cls(start=start.strip(), end=end.strip())
        if isinstance(raw, dict):
            return cls(
                start=str(raw["start"]),
                end=str(raw["end"]),
                name=str(raw.get("name") or ""),
                days=normalize_weekdays(raw.get("days") or raw.get("weekdays") or ()),
                enabled=bool(raw.get("enabled", True)),
            )
        raise ValueError(f"invalid time window: {raw!r}")

    def asdict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "days": list(self.days),
            "enabled": self.enabled,
        }


@dataclass
class PolicyConfig:
    timezone: str = "Europe/Madrid"
    windows: list[TimeWindow] = field(default_factory=list)
    max_troiani_gpus: int = 4
    max_jobs: int = 4
    max_gpus_per_job: int = 4
    max_runtime_s: int = 0
    preempt_grace_s: int = 120
    cooldown_s: int = 30
    memory_busy_gb: float = 2.0
    worker_heartbeat_s: int = 10
    worker_stale_s: int = 45
    control_lost_grace_s: int = 90
    auto_rollback: bool = False
    max_rollbacks: int = 2
    stall_seconds: int = 600
    loss_spike_factor: float = 8.0
    throughput_collapse_factor: float = 0.5
    underutil_threshold: float = 5.0
    stop_all: bool = False
    aggressiveness: str = "mid"
    retention: RetentionConfig = field(default_factory=RetentionConfig)


@dataclass
class ControlConfig:
    host: str = "127.0.0.1"
    port: int = 8787
    public_url: str = "http://127.0.0.1:8787"
    token: str = ""
    bind_mlflow: bool = True
    mlflow_port: int = 5000
    tensorboard_port: int = 6006


@dataclass
class PathsConfig:
    root: Path = Path("var")
    store: Path = Path("var/platform.db")
    checkpoints: Path = Path("var/checkpoints")
    artifacts: Path = Path("var/artifacts")
    datasets: Path = Path("var/datasets")
    mlflow: Path = Path("var/mlruns")
    mlflow_db: Path = Path("var/mlflow.db")
    tensorboard: Path = Path("var/tensorboard")
    logs: Path = Path("var/logs")

    def resolve(self, base: Path) -> "PathsConfig":
        def _abs(path: Path) -> Path:
            return path if path.is_absolute() else (base / path)

        return replace(
            self,
            root=_abs(self.root),
            store=_abs(self.store),
            checkpoints=_abs(self.checkpoints),
            artifacts=_abs(self.artifacts),
            datasets=_abs(self.datasets),
            mlflow=_abs(self.mlflow),
            mlflow_db=_abs(self.mlflow_db),
            tensorboard=_abs(self.tensorboard),
            logs=_abs(self.logs),
        )


@dataclass
class PlatformConfig:
    control: ControlConfig = field(default_factory=ControlConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    owned_users: tuple[str, ...] = ()
    nodes: dict[str, dict[str, str]] = field(default_factory=dict)
    source: Path | None = None

    def ensure_dirs(self) -> None:
        for path in (
            self.paths.root,
            self.paths.checkpoints,
            self.paths.artifacts,
            self.paths.datasets,
            self.paths.mlflow,
            self.paths.tensorboard,
            self.paths.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.paths.store.parent.mkdir(parents=True, exist_ok=True)
        self.paths.mlflow_db.parent.mkdir(parents=True, exist_ok=True)

    @property
    def mlflow_tracking_uri(self) -> str:
        from troiani_platform.tracking.uri import mlflow_tracking_uri

        return f"sqlite:///{self.paths.mlflow_db.resolve()}" if self.paths.mlflow_db else mlflow_tracking_uri(self.paths.mlflow)


def _as_path(value: Any, default: Path) -> Path:
    return Path(str(value)) if value else default


def load_config(path: str | Path | None = None) -> PlatformConfig:
    env_path = os.environ.get("TROIANI_PLATFORM_CONFIG")
    chosen = Path(path or env_path or _repo_root() / "config" / "platform.yaml")
    raw: dict[str, Any] = {}
    if chosen.is_file():
        raw = yaml.safe_load(chosen.read_text()) or {}

    control_raw = raw.get("control") or {}
    token = os.environ.get("TROIANI_PLATFORM_TOKEN", control_raw.get("token") or "")
    control = ControlConfig(
        host=str(control_raw.get("host", "127.0.0.1")),
        port=int(control_raw.get("port", 8787)),
        public_url=str(
            os.environ.get("TROIANI_PLATFORM_URL")
            or control_raw.get("public_url")
            or "http://127.0.0.1:8787"
        ),
        token=str(token or ""),
        bind_mlflow=bool(control_raw.get("bind_mlflow", True)),
        mlflow_port=int(control_raw.get("mlflow_port", 5000)),
        tensorboard_port=int(control_raw.get("tensorboard_port", 6006)),
    )

    paths_raw = raw.get("paths") or {}
    paths = PathsConfig(
        root=_as_path(paths_raw.get("root"), Path("var")),
        store=_as_path(paths_raw.get("store"), Path("var/platform.db")),
        checkpoints=_as_path(paths_raw.get("checkpoints"), Path("var/checkpoints")),
        artifacts=_as_path(paths_raw.get("artifacts"), Path("var/artifacts")),
        datasets=_as_path(paths_raw.get("datasets"), Path("var/datasets")),
        mlflow=_as_path(paths_raw.get("mlflow"), Path("var/mlruns")),
        mlflow_db=_as_path(paths_raw.get("mlflow_db"), Path("var/mlflow.db")),
        tensorboard=_as_path(paths_raw.get("tensorboard"), Path("var/tensorboard")),
        logs=_as_path(paths_raw.get("logs"), Path("var/logs")),
    ).resolve(chosen.parent.parent if chosen.is_file() else Path.cwd())

    policy_raw = raw.get("policy") or {}
    retention_raw = policy_raw.get("retention") or {}
    windows = [TimeWindow.from_mapping(item) for item in (policy_raw.get("windows") or [])]
    policy = PolicyConfig(
        timezone=str(policy_raw.get("timezone", "Europe/Madrid")),
        windows=windows,
        max_troiani_gpus=int(policy_raw.get("max_troiani_gpus", 4)),
        max_jobs=int(policy_raw.get("max_jobs", 4)),
        max_gpus_per_job=int(policy_raw.get("max_gpus_per_job", 4)),
        max_runtime_s=int(policy_raw.get("max_runtime_s", 0)),
        preempt_grace_s=int(policy_raw.get("preempt_grace_s", 120)),
        cooldown_s=int(policy_raw.get("cooldown_s", 30)),
        memory_busy_gb=float(policy_raw.get("memory_busy_gb", 2.0)),
        worker_heartbeat_s=int(policy_raw.get("worker_heartbeat_s", 10)),
        worker_stale_s=int(policy_raw.get("worker_stale_s", 45)),
        control_lost_grace_s=int(policy_raw.get("control_lost_grace_s", 90)),
        auto_rollback=bool(policy_raw.get("auto_rollback", False)),
        max_rollbacks=int(policy_raw.get("max_rollbacks", 2)),
        stall_seconds=int(policy_raw.get("stall_seconds", 600)),
        loss_spike_factor=float(policy_raw.get("loss_spike_factor", 8.0)),
        throughput_collapse_factor=float(policy_raw.get("throughput_collapse_factor", 0.5)),
        underutil_threshold=float(policy_raw.get("underutil_threshold", 5.0)),
        stop_all=bool(policy_raw.get("stop_all", False)),
        aggressiveness=str(policy_raw.get("aggressiveness") or "mid").lower(),
        retention=RetentionConfig(
            last_regular=int(retention_raw.get("last_regular", 5)),
            last_milestone=int(retention_raw.get("last_milestone", 3)),
            best_n=int(retention_raw.get("best_n", 3)),
            keep_latest_preemption=bool(retention_raw.get("keep_latest_preemption", True)),
            keep_latest_healthy=bool(retention_raw.get("keep_latest_healthy", True)),
        ),
    )

    owned = tuple(str(x) for x in (raw.get("owned_users") or []))
    nodes = {str(k): {str(a): str(b) for a, b in (v or {}).items()} for k, v in (raw.get("nodes") or {}).items()}
    cfg = PlatformConfig(
        control=control,
        paths=paths,
        policy=policy,
        owned_users=owned,
        nodes=nodes,
        source=chosen if chosen.is_file() else None,
    )
    return cfg
