from __future__ import annotations

from typing import Any, Iterable, Protocol, Sequence

from troiani_platform.models import CheckpointRecord, GPUResource, Job, Run, SchedulerDecision


class ResourceProvider(Protocol):
    def snapshot(self) -> Sequence[GPUResource]: ...


class Scheduler(Protocol):
    def schedule(
        self,
        jobs: Sequence[Job],
        gpus: Sequence[GPUResource],
        **kwargs: Any,
    ) -> list[SchedulerDecision]: ...


class JobExecutor(Protocol):
    def start(self, job: Job, run: Run) -> int: ...
    def request_checkpoint(self, job: Job) -> None: ...
    def stop(self, job: Job, graceful: bool = True) -> None: ...


class CheckpointStore(Protocol):
    def put(self, record: CheckpointRecord, payload: dict[str, bytes]) -> CheckpointRecord: ...
    def get(self, checkpoint_id: str) -> CheckpointRecord: ...
    def list(self, run_id: str) -> Iterable[CheckpointRecord]: ...


class ArtifactStore(Protocol):
    def put(self, run_id: str, name: str, data: bytes, meta: dict[str, Any]) -> str: ...
    def get(self, artifact_id: str) -> bytes: ...


class PolicyEngine(Protocol):
    def allows_admission(self, job: Job, running: Sequence[Job], gpus: Sequence[GPUResource], now: Any) -> Any: ...


class RunStore(Protocol):
    def put_run(self, run: Run) -> Run: ...
    def get_run(self, run_id: str) -> Run: ...


class MetricsSink(Protocol):
    def log(self, run_id: str, name: str, value: float, step: int | None = None) -> None: ...
