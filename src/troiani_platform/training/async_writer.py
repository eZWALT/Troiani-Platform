from __future__ import annotations

import copy
from multiprocessing import Process
from typing import Any, Callable

from troiani_platform.training.checkpoint import CheckpointManager, SaveRequest


def _child_save(root: str, request: SaveRequest, faults: set[str]) -> None:
    manager = CheckpointManager(root)
    manager.faults = set(faults)
    manager.save(request)


class AsyncCheckpointWriter:
    """Stage on the caller, persist in another process (own GIL). One in-flight save."""

    def __init__(self, manager: CheckpointManager) -> None:
        self.manager = manager
        self._proc: Process | None = None
        self._last: dict[str, Any] | None = None

    def submit(self, request: SaveRequest, on_done: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.wait()
        staged = SaveRequest(
            run_id=request.run_id,
            job_id=request.job_id,
            step=request.step,
            epoch=request.epoch,
            kind=request.kind,
            state=copy.deepcopy(request.state),
            artifacts=dict(request.artifacts) if request.artifacts else None,
            metadata=dict(request.metadata) if request.metadata else None,
            healthy=request.healthy,
            metric=request.metric,
            experiment=request.experiment,
        )
        # Emergency / last checkpoint stays synchronous so resume has a VALID dir.
        if str(request.kind.value) in {"preemption", "shutdown", "failure"}:
            self._last = self.manager.save(staged)
            if on_done and self._last:
                on_done(self._last)
            return
        self._proc = Process(target=_child_save, args=(str(self.manager.root), staged, set(self.manager.faults)))
        self._proc.start()

    def wait(self) -> None:
        if self._proc is not None:
            self._proc.join()
            self._proc = None
