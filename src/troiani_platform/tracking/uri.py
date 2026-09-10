from __future__ import annotations

from pathlib import Path


def mlflow_db_path(mlflow_dir: Path) -> Path:
    """SQLite tracking DB sits beside the artifact directory."""
    return Path(mlflow_dir).resolve().parent / "mlflow.db"


def mlflow_tracking_uri(mlflow_dir: Path) -> str:
    """MLflow 3 refuses the file store unless opted in. Use SQLite by default."""
    return f"sqlite:///{mlflow_db_path(mlflow_dir)}"


def mlflow_artifact_root(mlflow_dir: Path) -> str:
    path = Path(mlflow_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path.as_uri()
