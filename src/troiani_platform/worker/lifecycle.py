from __future__ import annotations

import os
import signal
from typing import Iterable


def request_checkpoint(pid: int) -> bool:
    """Ask a trainer for an emergency checkpoint. Missing PIDs are success: nothing left to signal."""
    try:
        os.kill(pid, signal.SIGUSR1)
        return True
    except ProcessLookupError:
        return False


def terminate(pid: int, graceful: bool = True) -> bool:
    try:
        os.kill(pid, signal.SIGTERM if graceful else signal.SIGKILL)
        return True
    except ProcessLookupError:
        return False


def visible_devices(local_indexes: Iterable[int]) -> str:
    return ",".join(str(i) for i in local_indexes)
