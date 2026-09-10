from __future__ import annotations

from typing import Any

from troiani_platform.models import Run


def reproduce_run(run: Run) -> dict[str, Any]:
    return {
        "run_id": run.id,
        "experiment": run.experiment,
        "git_sha": run.git_sha,
        "config": run.config,
        "environment": run.environment,
        "hardware": run.hardware,
        "dataset": run.dataset,
        "note": "Bit-for-bit determinism is not promised; this is enough to rerun the job.",
        "command": run.config.get("command"),
    }
