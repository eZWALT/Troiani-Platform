from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from troiani_platform.config import RetentionConfig
from troiani_platform.errors import CheckpointError
from troiani_platform.infra.notation import checkpoint_id, experiment_slug
from troiani_platform.models import CheckpointKind, CheckpointState

MANIFEST_VERSION = 1


def manifest_to_report(manifest: Mapping[str, Any], path: str | Path = "") -> dict[str, Any]:
    """Flatten a worker-local save manifest into a control-plane checkpoint report."""
    ckpt = dict(manifest.get("checkpoint") or {})
    run = dict(manifest.get("run") or {})
    integrity = dict(manifest.get("integrity") or {})
    ckpt_id = str(ckpt.get("id") or manifest.get("id") or "")
    run_id = str(run.get("id") or manifest.get("run_id") or "")
    return {
        "id": ckpt_id,
        "run_id": run_id,
        "job_id": str(run.get("job_id") or manifest.get("job_id") or ""),
        "step": int(ckpt.get("step") if ckpt.get("step") is not None else manifest.get("step") or 0),
        "kind": str(manifest.get("kind") or CheckpointKind.REGULAR.value),
        "healthy": bool(manifest.get("healthy")),
        "checksum": str(integrity.get("checksum") or manifest.get("checksum") or ""),
        "path": str(path or manifest.get("path") or ""),
    }


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_state(state: Any) -> bytes:
    if isinstance(state, (bytes, bytearray)):
        return bytes(state)
    try:
        import io

        import torch

        buffer = io.BytesIO()
        torch.save(state, buffer)
        return buffer.getvalue()
    except Exception:
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)


def load_state(data: bytes) -> Any:
    try:
        import io

        import torch

        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    except Exception:
        return pickle.loads(data)


@dataclass
class SaveRequest:
    run_id: str
    job_id: str
    step: int
    epoch: int = 0
    kind: CheckpointKind = CheckpointKind.REGULAR
    state: Any = None
    artifacts: Mapping[str, bytes] | None = None
    metadata: Mapping[str, Any] | None = None
    healthy: bool = False
    metric: float | None = None
    experiment: str | None = None


class CheckpointManager:
    def __init__(self, root: str | Path, retention: RetentionConfig | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.retention = retention or RetentionConfig()
        self.faults: set[str] = set()

    def run_dir(self, run_id: str) -> Path:
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _allocate_id(self, step: int, kind: CheckpointKind | str = CheckpointKind.REGULAR, experiment: str | None = None) -> str:
        if experiment:
            return checkpoint_id(experiment, step, str(kind.value if isinstance(kind, CheckpointKind) else kind))
        return f"ckpt-{step:06d}"

    def _index_experiment(self, experiment: str | None, run_id: str) -> None:
        if not experiment:
            return
        dest = self.root / "experiments" / experiment_slug(experiment) / run_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        target = self.root / run_id
        if dest.exists() or dest.is_symlink():
            return
        try:
            dest.symlink_to(target, target_is_directory=True)
        except OSError:
            dest.write_text(str(target) + "\n", encoding="utf-8")

    def save(self, request: SaveRequest) -> dict[str, Any]:
        if "disk_full" in self.faults:
            raise CheckpointError("disk full (fault injection)")
        if "slow_checkpoint" in self.faults:
            import time

            time.sleep(0.05)

        ckpt_id = self._allocate_id(request.step, request.kind, request.experiment)
        final = self.run_dir(request.run_id) / ckpt_id
        tmp = Path(str(final) + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

        artifacts = dict(request.artifacts or {})
        if request.state is not None:
            artifacts.setdefault("state.pkl", dump_state(request.state))

        checksums: dict[str, str] = {}
        for name, blob in artifacts.items():
            if "/" in name or name.startswith("."):
                raise CheckpointError(f"unsafe artifact name: {name}")
            _write_bytes(tmp / name, blob)
            checksums[name] = sha256_bytes(blob)

        if "corrupt_write" in self.faults:
            checksums["state.pkl"] = "deadbeef"

        created = datetime.now(timezone.utc).isoformat()
        experiment = request.experiment
        manifest = {
            "checkpoint": {"id": ckpt_id, "step": request.step, "epoch": request.epoch, "created_at": created},
            "run": {"id": request.run_id, "job_id": request.job_id, "experiment": experiment},
            "kind": request.kind.value,
            "healthy": request.healthy,
            "metric": request.metric,
            "training": dict((request.metadata or {}).get("training") or {}),
            "model": dict((request.metadata or {}).get("model") or {}),
            "data": dict((request.metadata or {}).get("data") or {}),
            "environment": dict((request.metadata or {}).get("environment") or {}),
            "artifacts": {name: checksums[name] for name in artifacts},
            "integrity": {"checksum": sha256_bytes(json.dumps(checksums, sort_keys=True).encode()), "version": MANIFEST_VERSION},
            "metadata": dict(request.metadata or {}),
        }
        _write_bytes(tmp / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode())
        (tmp / "VALIDATING").write_text("1")
        _fsync_dir(tmp)

        if "partial_write" in self.faults:
            raise CheckpointError("partial write (fault injection)")

        self._validate_dir(tmp, expect_artifacts=set(artifacts))
        if "corrupt_after_validate" in self.faults:
            (tmp / "state.pkl").write_bytes(b"truncated")

        if final.exists():
            shutil.rmtree(final)
        os.rename(tmp, final)
        _fsync_dir(final.parent)
        (final / "VALID").write_text("1")
        if (final / "VALIDATING").exists():
            (final / "VALIDATING").unlink()

        self._index_experiment(experiment, request.run_id)
        try:
            self.gc(request.run_id)
        except CheckpointError:
            pass
        return manifest

    def _validate_dir(self, path: Path, expect_artifacts: set[str] | None = None) -> dict[str, Any]:
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise CheckpointError(f"missing manifest in {path}")
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise CheckpointError("corrupt manifest") from exc
        artifacts = manifest.get("artifacts") or {}
        expected = expect_artifacts if expect_artifacts is not None else set(artifacts)
        for name in expected:
            file_path = path / name
            if not file_path.exists():
                raise CheckpointError(f"missing artifact {name}")
            digest = sha256_file(file_path)
            if artifacts.get(name) and artifacts[name] != digest:
                raise CheckpointError(f"checksum mismatch for {name}")
        if "state.pkl" in artifacts:
            data = (path / "state.pkl").read_bytes()
            try:
                load_state(data)
            except Exception as exc:
                raise CheckpointError("state reload failed") from exc
        return manifest

    def validate(self, run_id: str, checkpoint_id: str) -> dict[str, Any]:
        path = self.run_dir(run_id) / checkpoint_id
        if not path.exists():
            raise CheckpointError(f"checkpoint {checkpoint_id} not found")
        return self._validate_dir(path)

    def list(self, run_id: str) -> list[dict[str, Any]]:
        items = []
        for path in sorted(self.run_dir(run_id).iterdir()) if self.run_dir(run_id).exists() else []:
            if not path.is_dir() or path.name.endswith(".tmp"):
                continue
            manifest_path = path / "manifest.json"
            if not manifest_path.exists() or not (path / "VALID").exists():
                continue
            try:
                items.append(json.loads(manifest_path.read_text()))
            except json.JSONDecodeError:
                continue
        items.sort(key=lambda item: int(item.get("checkpoint", {}).get("step") or 0))
        return items

    def latest_valid(self, run_id: str) -> dict[str, Any] | None:
        for item in reversed(self.list(run_id)):
            ckpt_id = item["checkpoint"]["id"]
            try:
                return self.validate(run_id, ckpt_id)
            except CheckpointError:
                continue
        return None

    def load(self, run_id: str, checkpoint_id: str | None = None) -> Any:
        if checkpoint_id:
            manifest = self.validate(run_id, checkpoint_id)
            path = self.run_dir(run_id) / checkpoint_id
        else:
            manifest = self.latest_valid(run_id)
            if not manifest:
                raise CheckpointError(f"no valid checkpoint for {run_id}")
            path = self.run_dir(run_id) / manifest["checkpoint"]["id"]
        return load_state((path / "state.pkl").read_bytes())

    def gc(self, run_id: str) -> list[str]:
        items = self.list(run_id)
        keep: set[str] = set()
        regular = [i for i in items if i.get("kind") == CheckpointKind.REGULAR.value]
        milestone = [i for i in items if i.get("kind") == CheckpointKind.MILESTONE.value]
        best = sorted(
            [i for i in items if i.get("metric") is not None],
            key=lambda i: float(i["metric"]),
        )[: self.retention.best_n]
        for group, n in ((regular, self.retention.last_regular), (milestone, self.retention.last_milestone)):
            for item in group[-n:]:
                keep.add(item["checkpoint"]["id"])
        for item in best:
            keep.add(item["checkpoint"]["id"])
        if self.retention.keep_latest_preemption:
            pre = [i for i in items if i.get("kind") == CheckpointKind.PREEMPTION.value]
            if pre:
                keep.add(pre[-1]["checkpoint"]["id"])
        if self.retention.keep_latest_healthy:
            healthy = [i for i in items if i.get("healthy")]
            if healthy:
                keep.add(healthy[-1]["checkpoint"]["id"])
        latest = self.latest_valid(run_id)
        if latest:
            keep.add(latest["checkpoint"]["id"])
        if not keep and items:
            keep.add(items[-1]["checkpoint"]["id"])
        deleted = []
        root = self.run_dir(run_id)
        for path in root.iterdir():
            if not path.is_dir() or path.name.endswith(".tmp"):
                if path.name.endswith(".tmp"):
                    shutil.rmtree(path, ignore_errors=True)
                continue
            if path.name not in keep:
                shutil.rmtree(path, ignore_errors=True)
                deleted.append(path.name)
        return deleted
