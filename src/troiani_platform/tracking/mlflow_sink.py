from __future__ import annotations

from pathlib import Path
from typing import Any

from troiani_platform.tracking.uri import mlflow_tracking_uri


class MLflowSink:
    """Best-effort MLflow writer. Missing mlflow must never break the control plane."""

    def __init__(self, tracking_uri: str | Path) -> None:
        if isinstance(tracking_uri, Path):
            self.tracking_uri = mlflow_tracking_uri(tracking_uri)
        elif str(tracking_uri).startswith(("sqlite:", "http:", "https:", "file:")):
            self.tracking_uri = str(tracking_uri)
        else:
            self.tracking_uri = mlflow_tracking_uri(Path(tracking_uri))
        self._mlflow = None
        self._run_ids: dict[str, str] = {}

    def _client(self):
        if self._mlflow is None:
            try:
                import mlflow

                mlflow.set_tracking_uri(self.tracking_uri)
                self._mlflow = mlflow
            except Exception:
                self._mlflow = False
        return self._mlflow if self._mlflow else None

    def start_run(self, run_id: str, experiment: str, params: dict[str, Any], tags: dict[str, str]) -> None:
        mlflow = self._client()
        if not mlflow:
            return
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_id) as active:
            self._run_ids[run_id] = active.info.run_id
            mlflow.log_params({k: str(v)[:250] for k, v in params.items() if v is not None})
            mlflow.set_tags({k: str(v)[:250] for k, v in tags.items()})

    def log_metrics(self, experiment: str, run_name: str, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow = self._client()
        if not mlflow or not metrics:
            return
        mlflow.set_experiment(experiment)
        mlflow_id = self._run_ids.get(run_name)
        ctx = mlflow.start_run(run_id=mlflow_id) if mlflow_id else mlflow.start_run(run_name=run_name)
        with ctx as active:
            self._run_ids[run_name] = active.info.run_id
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)
