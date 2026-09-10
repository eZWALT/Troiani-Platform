from __future__ import annotations

from troiani_platform.training.checkpoint import CheckpointManager


class FaultInjector:
    def __init__(self, manager: CheckpointManager | None = None) -> None:
        self.manager = manager

    def enable(self, name: str) -> None:
        if self.manager is not None:
            self.manager.faults.add(name)

    def disable(self, name: str) -> None:
        if self.manager is not None:
            self.manager.faults.discard(name)
