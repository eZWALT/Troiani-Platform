from __future__ import annotations

import os
import socket


def hostname() -> str:
    return socket.gethostname().split(".")[0]


def current_node(explicit: str | None = None) -> str:
    return explicit or os.environ.get("TROIANI_NODE") or hostname()
