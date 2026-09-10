from __future__ import annotations

import os
import pwd
import shutil
import subprocess
from typing import Iterable

import psutil


def username_for_pid(pid: int) -> str | None:
    try:
        return psutil.Process(pid).username()
    except (psutil.Error, PermissionError):
        try:
            return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
        except OSError:
            return None


def command_for_pid(pid: int) -> str | None:
    try:
        proc = psutil.Process(pid)
        cmdline = proc.cmdline()
        return " ".join(cmdline) if cmdline else proc.name()
    except (psutil.Error, PermissionError):
        return None


def classify_process(username: str | None, command: str | None, owned_users: Iterable[str]) -> bool:
    owned = {u.lower() for u in owned_users}
    if username and username.lower() in owned:
        cmd = (command or "").lower()
        markers = ("troiani_platform", "troiani-platform", "TROIANI_JOB_ID=", "dummy_train")
        if any(marker.lower() in cmd for marker in markers):
            return True
        if os.environ.get("TROIANI_JOB_ID") and username.lower() in owned:
            return "troiani" in cmd
    return False


def list_logins() -> list[dict[str, str]]:
    who = shutil.which("who")
    if not who:
        return []
    try:
        out = subprocess.check_output([who], text=True, timeout=3)
    except (subprocess.SubprocessError, OSError):
        return []
    sessions = []
    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        sessions.append({"user": parts[0], "raw": line})
    return sessions
