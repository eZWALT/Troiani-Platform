from __future__ import annotations

import shlex
from typing import Any

from troiani_platform.errors import JobSpecError


def _accelerators(raw: Any) -> tuple[int, str | None, float]:
    if raw is None:
        return 1, None, 0.0
    if isinstance(raw, int):
        return max(1, raw), None, 0.0
    text = str(raw).strip()
    # SkyPilot: "A100:2" or "A100-80GB:1"
    if ":" in text:
        kind, count = text.rsplit(":", 1)
        try:
            gpus = max(1, int(count))
        except ValueError:
            gpus = 1
        token = kind.strip()
        memory = 80.0 if "80" in token else 40.0 if "40" in token else 0.0
        return gpus, token or None, memory
    return 1, text or None, 0.0


def _command(raw: dict[str, Any]) -> list[str]:
    if raw.get("command"):
        command = raw["command"]
        if isinstance(command, str):
            parts = shlex.split(command)
            if not parts:
                raise JobSpecError("command string is empty")
            return parts
        return [str(x) for x in command]
    if raw.get("entrypoint"):
        entry = raw["entrypoint"]
        args = raw.get("args") or []
        if isinstance(entry, str):
            parts = shlex.split(entry)
        else:
            parts = [str(x) for x in entry]
        if isinstance(args, str):
            parts.extend(shlex.split(args))
        else:
            parts.extend(str(x) for x in args)
        if not parts:
            raise JobSpecError("entrypoint is empty")
        return parts
    run = raw.get("run")
    if isinstance(run, str) and run.strip():
        # Tokenize only. Never shell=True.
        first = run.strip().splitlines()[0].strip()
        parts = shlex.split(first)
        if not parts:
            raise JobSpecError("run string is empty")
        return parts
    if isinstance(run, list) and run:
        return [str(x) for x in run]
    raise JobSpecError("need command, entrypoint, or run")


def spec_from_task_yaml(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept Troiani job YAML or a SkyPilot-shaped task file."""
    if not isinstance(raw, dict):
        raise JobSpecError("YAML root must be a mapping")
    resources = raw.get("resources") if isinstance(raw.get("resources"), dict) else {}
    gpus, preferred, memory = _accelerators(resources.get("accelerators") or raw.get("accelerators"))
    if raw.get("gpus"):
        gpus = int(raw["gpus"])
    if raw.get("preferred_gpu_type"):
        preferred = str(raw["preferred_gpu_type"])
    if raw.get("min_gpu_memory_gb"):
        memory = float(raw["min_gpu_memory_gb"])
    name = str(raw.get("name") or raw.get("experiment") or "adhoc")
    workdir = raw.get("workdir") or raw.get("cwd")
    spec = {
        "name": name.replace(" ", "-")[:128],
        "experiment": raw.get("experiment") or name,
        "command": _command(raw),
        "gpus": gpus,
        "min_gpu_memory_gb": memory,
        "preferred_gpu_type": preferred,
        "preferred_node": str(raw.get("preferred_node") or (resources.get("infra") if isinstance(resources.get("infra"), str) else "") or ""),
        "priority": raw.get("priority") or "opportunistic",
        "preemptible": bool(raw.get("preemptible", True)),
        "checkpoint_interval": int(raw.get("checkpoint_interval") or 600),
        "checkpoint_every_steps": raw.get("checkpoint_every_steps"),
        "max_runtime": int(raw.get("max_runtime") or 0),
        "cwd": str(workdir) if workdir else None,
        "env": dict(raw.get("envs") or raw.get("env") or {}),
        "dataset": raw.get("dataset"),
        "evaluation": raw.get("evaluation"),
    }
    if spec["preferred_node"] in {"aws", "gcp", "azure"}:
        spec["preferred_node"] = ""
    return spec
