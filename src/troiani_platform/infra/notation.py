from __future__ import annotations

import re

_SLUG = re.compile(r"[^a-z0-9]+")
_KIND_TAG = {
    "regular": "ckpt",
    "milestone": "mile",
    "preemption": "preempt",
    "best": "best",
    "healthy": "healthy",
    "shutdown": "stop",
    "failure": "fail",
    "eval": "eval",
}


def experiment_slug(name: str | None, fallback: str = "exp") -> str:
    text = _SLUG.sub("-", (name or fallback).strip().lower()).strip("-")
    return (text or fallback)[:48]


def gpu_ref(node: str | None, index: int | None, uuid: str | None = None) -> str:
    host = (node or "node").strip().lower() or "node"
    if index is None and uuid:
        return f"{host}/{uuid[:8]}"
    return f"{host}/gpu{int(index or 0)}"


def checkpoint_id(experiment: str | None, step: int, kind: str = "regular") -> str:
    tag = _KIND_TAG.get(str(kind).lower(), "ckpt")
    return f"{experiment_slug(experiment)}-{tag}-{int(step):06d}"
