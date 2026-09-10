from __future__ import annotations

from pathlib import Path
from typing import Any

from troiani_platform.config import PathsConfig


def _dir_bytes(path: Path) -> tuple[int, int]:
    total = 0
    files = 0
    if not path.exists():
        return 0, 0
    if path.is_file():
        return path.stat().st_size, 1
    for item in path.rglob("*"):
        if item.is_file():
            files += 1
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total, files


def _gb(value: int) -> float:
    return round(value / (1024**3), 3)


def storage_report(paths: PathsConfig) -> dict[str, Any]:
    roots = {
        "root": paths.root,
        "checkpoints": paths.checkpoints,
        "artifacts": paths.artifacts,
        "logs": paths.logs,
        "mlflow": paths.mlflow,
        "tensorboard": paths.tensorboard,
        "datasets": paths.datasets,
    }
    folders = []
    for name, path in roots.items():
        size, files = _dir_bytes(path)
        folders.append({"name": name, "path": str(path), "bytes": size, "gb": _gb(size), "files": files})
    disk = None
    try:
        usage = __import__("shutil").disk_usage(paths.root)
        disk = {"total_gb": _gb(usage.total), "used_gb": _gb(usage.used), "free_gb": _gb(usage.free)}
    except OSError:
        disk = None
    runs: list[dict[str, Any]] = []
    ckpt_root = paths.checkpoints
    if ckpt_root.is_dir():
        for run_dir in sorted(p for p in ckpt_root.iterdir() if p.is_dir()):
            size, files = _dir_bytes(run_dir)
            ckpts = [p.name for p in run_dir.iterdir() if p.is_dir() and not p.name.endswith(".tmp")]
            runs.append(
                {
                    "run_id": run_dir.name,
                    "path": str(run_dir),
                    "bytes": size,
                    "gb": _gb(size),
                    "checkpoints": len(ckpts),
                    "names": ckpts[:12],
                }
            )
    runs.sort(key=lambda row: row["bytes"], reverse=True)
    return {"disk": disk, "folders": folders, "checkpoint_runs": runs[:40]}


def node_snapshot(paths: PathsConfig | None = None) -> dict[str, Any]:
    """Cheap per-node disk + process view for worker heartbeats."""
    root = paths.root if paths else Path("var")
    report: dict[str, Any] = {"top": [], "disk": None, "checkpoints": None}
    try:
        usage = __import__("shutil").disk_usage(root)
        report["disk"] = {"total_gb": _gb(usage.total), "used_gb": _gb(usage.used), "free_gb": _gb(usage.free)}
    except OSError:
        pass
    if paths:
        size, files = _dir_bytes(paths.checkpoints)
        report["checkpoints"] = {"path": str(paths.checkpoints), "bytes": size, "gb": _gb(size), "files": files}
        report["folders"] = storage_report(paths)["folders"]
    try:
        import psutil

        rows = []
        for proc in psutil.process_iter(["pid", "username", "name", "cpu_percent", "memory_info"]):
            try:
                info = proc.info
                rss = float((info.get("memory_info").rss if info.get("memory_info") else 0) / (1024**2))
                rows.append(
                    {
                        "pid": info.get("pid"),
                        "user": info.get("username"),
                        "name": (info.get("name") or "")[:40],
                        "cpu": float(info.get("cpu_percent") or 0),
                        "rss_mb": round(rss, 1),
                    }
                )
            except (psutil.Error, OSError, AttributeError):
                continue
        rows.sort(key=lambda row: (-row["cpu"], -row["rss_mb"]))
        report["top"] = rows[:18]
    except Exception:
        report["top"] = []
    return report
