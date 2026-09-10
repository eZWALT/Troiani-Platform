"""Tiny cooperative trainer used for platform smoke tests. No GPU required."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

from troiani_platform.models import CheckpointKind
from troiani_platform.training.async_writer import AsyncCheckpointWriter
from troiani_platform.training.checkpoint import CheckpointManager, SaveRequest, load_state, manifest_to_report
from troiani_platform.training.eval_hook import decode_eval_env, due_this_step, run_eval_command
from troiani_platform.training.health import HealthMonitor

_preempt = False
_shutdown = False


def _handle_preempt(_signum: int, _frame: object | None) -> None:
    global _preempt
    _preempt = True


def _handle_stop(_signum: int, _frame: object | None) -> None:
    global _shutdown
    _shutdown = True


def _post(endpoint: str, path: str, payload: dict) -> None:
    if not endpoint:
        return
    try:
        import urllib.request

        req = urllib.request.Request(
            endpoint.rstrip("/") + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('TROIANI_PLATFORM_TOKEN', '')}"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:
        pass


def _report_checkpoint(endpoint: str, manager: CheckpointManager, manifest: dict) -> None:
    ckpt = manifest.get("checkpoint") or {}
    run = manifest.get("run") or {}
    path = ""
    if ckpt.get("id") and run.get("id"):
        path = str(manager.run_dir(str(run["id"])) / str(ckpt["id"]))
    _post(endpoint, "/v1/internal/checkpoint", manifest_to_report(manifest, path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--fail-at", type=int, default=0)
    parser.add_argument("--nan-at", type=int, default=0)
    parser.add_argument("--inject-nan", action="store_true")
    parser.add_argument("--tokens-per-step", type=int, default=2048)
    args = parser.parse_args()

    signal.signal(signal.SIGUSR1, _handle_preempt)
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    run_id = os.environ.get("RUN_ID", "run-local")
    job_id = os.environ.get("JOB_ID", "job-local")
    experiment = os.environ.get("EXPERIMENT") or os.environ.get("JOB_NAME") or None
    ckpt_dir = Path(os.environ.get("CHECKPOINT_DIR", "var/checkpoints"))
    endpoint = os.environ.get("PLATFORM_ENDPOINT", "")
    manager = CheckpointManager(ckpt_dir)
    writer = AsyncCheckpointWriter(manager)
    health = HealthMonitor(stall_seconds=30)

    def _save(step: int, kind: CheckpointKind, state: dict, **kwargs) -> dict | None:
        request = SaveRequest(
            run_id=run_id,
            job_id=job_id,
            step=step,
            kind=kind,
            state=state,
            experiment=experiment,
            **kwargs,
        )
        if kind in {CheckpointKind.PREEMPTION, CheckpointKind.SHUTDOWN, CheckpointKind.FAILURE}:
            writer.wait()
            manifest = manager.save(request)
            _report_checkpoint(endpoint, manager, manifest)
            return manifest
        writer.submit(request)
        return None

    step = 0
    rng_state = 0.0
    latest = manager.latest_valid(run_id)
    if latest:
        state = load_state((manager.run_dir(run_id) / latest["checkpoint"]["id"] / "state.pkl").read_bytes())
        step = int(state.get("step", 0))
        rng_state = float(state.get("rng", 0.0))

    start = time.time()
    try:
        hook = decode_eval_env(os.environ)
    except Exception:
        hook = None
    while step < args.steps and not _shutdown:
        if _preempt:
            break
        step += 1
        rng_state += 0.01
        loss = 1.0 / math.sqrt(step + 1) + 0.01 * math.sin(step / 7)
        if args.inject_nan or (args.nan_at and step == args.nan_at):
            loss = float("nan")
        if args.fail_at and step == args.fail_at:
            raise RuntimeError("injected training crash")
        throughput = 1000.0 / max(args.sleep, 1e-6)
        tokens_delta = max(1, int(args.tokens_per_step))
        tokens_ingested = step * tokens_delta
        tokens_per_sec = tokens_delta / max(args.sleep, 1e-6)
        anomalies = health.observe(ts=time.time(), loss=loss, step=step, throughput=throughput)
        _post(
            endpoint,
            "/v1/internal/metrics",
            {
                "job_id": job_id,
                "run_id": run_id,
                "step": step,
                "loss": loss if loss == loss else None,
                "throughput": throughput,
                "tokens_delta": tokens_delta,
                "tokens_ingested": tokens_ingested,
                "tokens_per_sec": tokens_per_sec,
                "anomalies": [a.value for a in anomalies],
            },
        )
        if hook and due_this_step(step, hook[0]):
            try:
                result = run_eval_command(
                    hook[1],
                    env={**os.environ, "EVAL_STEP": str(step)},
                )
            except Exception:
                result = {"ok": False, "metrics": {}}
            eval_metrics = result.get("metrics") or {}
            eval_payload = {
                "job_id": job_id,
                "run_id": run_id,
                "step": step,
                "loss": loss if loss == loss else None,
                "val_loss": eval_metrics.get("val_loss"),
                "eval_ok": bool(result.get("ok")),
            }
            _post(endpoint, "/v1/internal/metrics", eval_payload)
        print(
            f"step={step} loss={loss:.4f} tokens={tokens_ingested} tok/s={tokens_per_sec:.0f}",
            flush=True,
        )
        if anomalies:
            _save(
                step,
                CheckpointKind.FAILURE,
                {"step": step, "rng": rng_state, "loss": None},
                metadata={"training": {"reason": "anomaly"}},
                healthy=False,
            )
            _post(endpoint, "/v1/internal/anomaly", {"job_id": job_id, "run_id": run_id, "kinds": [a.value for a in anomalies]})
            return
        interval = int(os.environ.get("CHECKPOINT_EVERY_STEPS") or 25)
        if interval and step % interval == 0:
            _save(
                step,
                CheckpointKind.REGULAR,
                {"step": step, "rng": rng_state, "loss": loss},
                metadata={"training": {"optimizer": "sgd"}},
                healthy=True,
                metric=loss,
            )
        time.sleep(args.sleep)

    kind = CheckpointKind.PREEMPTION if _preempt else CheckpointKind.SHUTDOWN
    _save(
        step,
        kind,
        {"step": step, "rng": rng_state},
        metadata={"training": {"elapsed_s": time.time() - start}},
        healthy=not _preempt or step > 0,
    )
    done = manager.run_dir(run_id) / "DONE"
    done.write_text(json.dumps({"step": step, "preempt": _preempt, "shutdown": _shutdown}))
    if _preempt:
        print(f"preempted at step={step}", flush=True)
        sys.exit(75)
    print(f"completed step={step}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
