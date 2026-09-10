from troiani_platform.tracking.mlflow_sink import MLflowSink
from troiani_platform.tracking.uri import mlflow_artifact_root, mlflow_db_path, mlflow_tracking_uri


def test_sqlite_uri(tmp_path):
    root = tmp_path / "mlruns"
    uri = mlflow_tracking_uri(root)
    assert uri.startswith("sqlite:///")
    assert uri.endswith("mlflow.db")
    assert mlflow_db_path(root) == tmp_path / "mlflow.db"
    assert mlflow_artifact_root(root).startswith("file:")
    assert root.is_dir()


def test_sink_accepts_directory_path(tmp_path):
    sink = MLflowSink(tmp_path / "mlruns")
    assert sink.tracking_uri.startswith("sqlite:///")


def test_sink_roundtrip_metrics(tmp_path):
    db = tmp_path / "mlflow.db"
    sink = MLflowSink(f"sqlite:///{db}")
    sink.start_run("run-test", "smoke", {"gpus": 1}, {"job_id": "job-1"})
    sink.log_metrics("smoke", "run-test", {"loss": 1.25}, step=3)
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{db}")
    runs = mlflow.search_runs(experiment_names=["smoke"])
    assert not runs.empty
    assert float(runs.iloc[0]["metrics.loss"]) == 1.25
