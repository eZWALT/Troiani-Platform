from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from troiani_platform.models import Artifact, new_id, utcnow


def checksum_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FileArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, run_id: str, name: str, data: bytes, meta: dict[str, Any] | None = None) -> Artifact:
        dest = self.root / run_id
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_bytes(data)
        return Artifact(
            id=new_id("art"),
            run_id=run_id,
            kind=str((meta or {}).get("kind") or "file"),
            path=str(path),
            checksum=checksum_bytes(data),
            meta=meta or {},
            created_at=utcnow(),
        )

    def get(self, artifact: Artifact) -> bytes:
        return Path(artifact.path).read_bytes()
