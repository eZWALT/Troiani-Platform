from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from troiani_platform.models import DatasetFingerprint


def hash_manifest(manifest: dict[str, Any]) -> str:
    blob = json.dumps(manifest, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def fingerprint_dataset(path: str | Path | None, fallback: DatasetFingerprint | None = None) -> DatasetFingerprint | None:
    if fallback and not path:
        if not fallback.dataset_hash and fallback.name:
            return DatasetFingerprint(
                name=fallback.name,
                version=fallback.version,
                revision=fallback.revision or hash_manifest(fallback.to_dict()),
                dataset_hash=fallback.dataset_hash or hash_manifest(fallback.to_dict()),
                shards=fallback.shards,
            )
        return fallback
    if not path:
        return fallback
    root = Path(path)
    manifest_path = root / "manifest.json" if root.is_dir() else root
    if not manifest_path.exists():
        return fallback
    manifest = json.loads(manifest_path.read_text())
    shards = tuple(manifest.get("shards") or [])
    digest = hash_manifest(manifest)
    revision_file = root / "revision" if root.is_dir() else None
    revision = revision_file.read_text().strip() if revision_file and revision_file.exists() else digest
    return DatasetFingerprint(
        name=str(manifest.get("name") or (fallback.name if fallback else root.name)),
        version=str(manifest.get("version") or (fallback.version if fallback else "0")),
        revision=revision,
        dataset_hash=digest,
        shards=shards,
    )


def write_dataset_manifest(root: str | Path, name: str, version: str, shards: list[str]) -> DatasetFingerprint:
    dest = Path(root)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"name": name, "version": version, "shards": shards}
    digest = hash_manifest(manifest)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (dest / "revision").write_text(digest)
    (dest / "shards").mkdir(exist_ok=True)
    return DatasetFingerprint(name=name, version=version, revision=digest, dataset_hash=digest, shards=tuple(shards))
