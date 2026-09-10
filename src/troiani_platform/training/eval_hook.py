"""Run JobSpec.evaluation commands as argv lists (never a shell)."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Mapping, Sequence


def encode_eval_env(every_steps: int, command: Sequence[str]) -> dict[str, str]:
    return {
        "EVAL_EVERY_STEPS": str(int(every_steps)),
        "EVAL_COMMAND": json.dumps([str(part) for part in command]),
    }


def decode_eval_env(env: Mapping[str, str]) -> tuple[int, list[str]] | None:
    raw_every = str(env.get("EVAL_EVERY_STEPS") or "").strip()
    raw_cmd = str(env.get("EVAL_COMMAND") or "").strip()
    if not raw_every or not raw_cmd:
        return None
    every = int(raw_every)
    if every <= 0:
        return None
    command = json.loads(raw_cmd)
    if isinstance(command, str):
        raise ValueError("evaluation.command must be a list, not a shell string")
    if not isinstance(command, list) or not command:
        return None
    return every, [str(part) for part in command]


def due_this_step(step: int, every_steps: int) -> bool:
    return every_steps > 0 and step > 0 and step % every_steps == 0


def normalize_argv(command: Sequence[str]) -> list[str]:
    argv = [str(part) for part in command]
    if not argv or any(not part for part in argv):
        raise ValueError("evaluation.command must be a non-empty argv list")
    if argv[0] in {"python", "python3"}:
        argv[0] = sys.executable
    return argv


def parse_eval_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def numeric_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if key in {"ok", "run_id", "job_id"} or value is None or isinstance(value, bool):
            continue
        try:
            metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def run_eval_command(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    argv = normalize_argv(command)
    completed = subprocess.run(
        argv,
        env=dict(env) if env is not None else None,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        shell=False,
    )
    parsed = parse_eval_stdout(completed.stdout)
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "metrics": numeric_metrics(parsed),
        "payload": parsed,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "argv": argv,
    }
