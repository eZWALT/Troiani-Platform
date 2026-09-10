from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    run_id = os.environ.get("RUN_ID", "run-local")
    step = int(os.environ.get("EVAL_STEP") or 0)
    out = Path(os.environ.get("CHECKPOINT_DIR", "var/checkpoints")) / run_id / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "step": step, "val_loss": 1.23, "ok": True}
    out.write_text(json.dumps(payload))
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
