from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from troiani_platform.infra.notation import gpu_ref
from troiani_platform.models import GPUResource, Occupancy


@dataclass
class InfraMetrics:
    gpu_seconds_total: float = 0.0
    troiani_gpu_seconds: float = 0.0
    researcher_gpu_seconds: float = 0.0
    available_gpu_seconds: float = 0.0
    preemptions: int = 0
    checkpoint_failures: int = 0
    tokens_ingested: float = 0.0
    checkpoint_duration_s: list[float] = field(default_factory=list)
    resume_latency_s: list[float] = field(default_factory=list)
    preemption_latency_s: list[float] = field(default_factory=list)
    util_history: list[dict] = field(default_factory=list)
    node_gpu_seconds: dict[str, float] = field(default_factory=dict)
    node_researcher_seconds: dict[str, float] = field(default_factory=dict)
    user_gpu_seconds: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)

    def observe_gpus(self, gpus: list[GPUResource], dt: float) -> None:
        seen: set[tuple[str, str, str]] = set()
        for gpu in gpus:
            self.gpu_seconds_total += dt
            node = gpu.node or "unknown"
            self.node_gpu_seconds[node] = self.node_gpu_seconds.get(node, 0.0) + dt
            if gpu.occupancy == Occupancy.TROIANI:
                self.troiani_gpu_seconds += dt
            elif gpu.occupancy == Occupancy.RESEARCHER:
                self.researcher_gpu_seconds += dt
                self.node_researcher_seconds[node] = self.node_researcher_seconds.get(node, 0.0) + dt
            elif gpu.occupancy == Occupancy.AVAILABLE:
                self.available_gpu_seconds += dt
            for proc in gpu.compute_processes:
                user = (proc.username or "").strip() or "?"
                key = (node, user, gpu.uuid)
                if key in seen:
                    continue
                seen.add(key)
                bucket = self.user_gpu_seconds.setdefault(node, {}).setdefault(user, {"seconds": 0.0, "troiani": 0.0, "other": 0.0})
                bucket["seconds"] += dt
                if proc.troiani_owned:
                    bucket["troiani"] += dt
                else:
                    bucket["other"] += dt

    def user_occupancy(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for node, users in self.user_gpu_seconds.items():
            for username, bucket in users.items():
                rows.append(
                    {
                        "node": node,
                        "username": username,
                        "seconds": float(bucket.get("seconds") or 0.0),
                        "troiani_s": float(bucket.get("troiani") or 0.0),
                        "other_s": float(bucket.get("other") or 0.0),
                    }
                )
        rows.sort(key=lambda row: (-row["seconds"], row["node"], row["username"]))
        return rows

    def occupancy_seconds(self) -> dict[str, dict[str, float]]:
        nodes = set(self.node_gpu_seconds) | set(self.node_researcher_seconds)
        return {
            node: {
                "researcher": float(self.node_researcher_seconds.get(node, 0.0)),
                "total": float(self.node_gpu_seconds.get(node, 0.0)),
            }
            for node in nodes
        }

    def observe_tokens(self, delta: float) -> None:
        if delta > 0:
            self.tokens_ingested += delta

    def observe_util(self, gpus: list[GPUResource], ts: str) -> None:
        sample = {"ts": ts}
        for gpu in gpus:
            key = f"{gpu.node}:{gpu.index}"
            sample[key] = {
                "sm": float(gpu.sm_util),
                "vram": round(gpu.vram_pct, 1),
                "mem_ctrl": gpu.memory_util,
                "util": float(gpu.sm_util),
                "mem": gpu.memory_used_gb,
                "occupancy": gpu.occupancy.value,
                "ref": gpu_ref(gpu.node, gpu.index, gpu.uuid),
            }
        self.util_history.append(sample)
        self.util_history = self.util_history[-180:]

    def snapshot(self) -> dict[str, Any]:
        def avg(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        useful = self.troiani_gpu_seconds
        overhead = sum(self.checkpoint_duration_s) + sum(self.preemption_latency_s)
        return {
            "gpu_seconds_total": self.gpu_seconds_total,
            "troiani_gpu_seconds": self.troiani_gpu_seconds,
            "researcher_gpu_seconds": self.researcher_gpu_seconds,
            "available_gpu_seconds": self.available_gpu_seconds,
            "preemptions": float(self.preemptions),
            "checkpoint_failures": float(self.checkpoint_failures),
            "checkpoint_duration_avg_s": avg(self.checkpoint_duration_s),
            "resume_latency_avg_s": avg(self.resume_latency_s),
            "preemption_latency_avg_s": avg(self.preemption_latency_s),
            "useful_training_time_s": useful,
            "checkpoint_overhead_s": overhead,
            "tokens_ingested": self.tokens_ingested,
            "history": list(self.util_history),
            "user_occupancy": self.user_occupancy(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_seconds_total": self.gpu_seconds_total,
            "troiani_gpu_seconds": self.troiani_gpu_seconds,
            "researcher_gpu_seconds": self.researcher_gpu_seconds,
            "available_gpu_seconds": self.available_gpu_seconds,
            "preemptions": self.preemptions,
            "checkpoint_failures": self.checkpoint_failures,
            "tokens_ingested": self.tokens_ingested,
            "node_gpu_seconds": dict(self.node_gpu_seconds),
            "node_researcher_seconds": dict(self.node_researcher_seconds),
            "user_gpu_seconds": self.user_gpu_seconds,
            "util_history": list(self.util_history[-60:]),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "InfraMetrics":
        metrics = cls()
        if not isinstance(raw, dict):
            return metrics
        for key in (
            "gpu_seconds_total",
            "troiani_gpu_seconds",
            "researcher_gpu_seconds",
            "available_gpu_seconds",
            "tokens_ingested",
        ):
            if raw.get(key) is not None:
                setattr(metrics, key, float(raw[key]))
        metrics.preemptions = int(raw.get("preemptions") or 0)
        metrics.checkpoint_failures = int(raw.get("checkpoint_failures") or 0)
        metrics.node_gpu_seconds = {str(k): float(v) for k, v in (raw.get("node_gpu_seconds") or {}).items()}
        metrics.node_researcher_seconds = {
            str(k): float(v) for k, v in (raw.get("node_researcher_seconds") or {}).items()
        }
        users: dict[str, dict[str, dict[str, float]]] = {}
        for node, by_user in (raw.get("user_gpu_seconds") or {}).items():
            users[str(node)] = {}
            for username, bucket in (by_user or {}).items():
                users[str(node)][str(username)] = {
                    "seconds": float((bucket or {}).get("seconds") or 0.0),
                    "troiani": float((bucket or {}).get("troiani") or 0.0),
                    "other": float((bucket or {}).get("other") or 0.0),
                }
        metrics.user_gpu_seconds = users
        history = raw.get("util_history") or []
        if isinstance(history, list):
            metrics.util_history = [item for item in history if isinstance(item, dict)][-180:]
        return metrics
