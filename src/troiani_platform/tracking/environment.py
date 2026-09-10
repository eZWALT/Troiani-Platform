from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def git_sha(cwd: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or os.getcwd(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def capture_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "executable": sys.executable,
    }
    try:
        import torch

        env["pytorch"] = torch.__version__
        env["cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        import importlib.metadata as md

        env["troiani_platform"] = md.version("troiani-platform")
    except Exception:
        env["troiani_platform"] = "0.1.0"
    return env
