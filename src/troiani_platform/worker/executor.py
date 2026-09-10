from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from troiani_platform.models import Job, Run
from troiani_platform.training.runner import build_env
from troiani_platform.worker.lifecycle import visible_devices


class ProcessExecutor:
    def start(
        self,
        job: Job,
        run: Run,
        checkpoint_dir: Path,
        endpoint: str,
        token: str,
        local_indexes: list[int],
    ) -> subprocess.Popen[bytes]:
        env = build_env(job, run, checkpoint_dir, endpoint, token)
        env["CUDA_VISIBLE_DEVICES"] = visible_devices(local_indexes)
        env["PYTHONUNBUFFERED"] = "1"
        command = list(job.spec.command)
        if command and command[0] in {"python", "python3"}:
            command[0] = sys.executable
        if command and Path(str(command[0])).name.startswith("python") and "-u" not in command:
            command = [command[0], "-u", *command[1:]]
        log_dir = Path(env["CHECKPOINT_DIR"]).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{job.id}.log"
        handle = log_path.open("ab", buffering=0)
        handle.write(f"# start {' '.join(command)}\n".encode())
        handle.flush()
        proc = subprocess.Popen(
            command,
            env=env,
            cwd=job.spec.cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        proc._troiani_log = handle  # type: ignore[attr-defined]
        proc._troiani_log_path = log_path  # type: ignore[attr-defined]
        return proc
