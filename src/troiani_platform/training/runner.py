from __future__ import annotations

import os
from pathlib import Path

from troiani_platform.models import Job, Run
from troiani_platform.training.eval_hook import encode_eval_env


def build_env(job: Job, run: Run, checkpoint_dir: Path, endpoint: str, token: str = "") -> dict[str, str]:
    env = os.environ.copy()
    env.update(job.spec.env)
    env["RUN_ID"] = run.id
    env["JOB_ID"] = job.id
    env["CHECKPOINT_DIR"] = str(checkpoint_dir)
    env["PLATFORM_ENDPOINT"] = endpoint
    env["GPU_ASSIGNMENT"] = ",".join(job.assigned_gpus)
    env["TROIANI_JOB_ID"] = job.id
    env["TROIANI_PLATFORM_TOKEN"] = token
    env["EXPERIMENT"] = str(job.spec.experiment or run.experiment or job.spec.name)
    env["JOB_NAME"] = job.spec.name
    if job.spec.checkpoint_every_steps:
        env["CHECKPOINT_EVERY_STEPS"] = str(job.spec.checkpoint_every_steps)
    if job.spec.evaluation:
        env.update(encode_eval_env(job.spec.evaluation.every_steps, job.spec.evaluation.command))
    if job.assigned_gpus:
        # CUDA_VISIBLE_DEVICES is filled by the worker from local indexes.
        env.setdefault("TROIANI_GPU_UUIDS", ",".join(job.assigned_gpus))
    return env
