from __future__ import annotations

from pathlib import Path

import pytest

from troiani_platform.config import PlatformConfig
from troiani_platform.control.service import PlatformService
from troiani_platform.models import JobSpec, Priority
from troiani_platform.state.store import StateStore


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> PlatformConfig:
    cfg = PlatformConfig()
    cfg.paths = cfg.paths.resolve(tmp_path)
    cfg.paths.root = tmp_path
    cfg.paths.store = tmp_path / "platform.db"
    cfg.paths.checkpoints = tmp_path / "ckpts"
    cfg.paths.artifacts = tmp_path / "arts"
    cfg.paths.datasets = tmp_path / "data"
    cfg.paths.mlflow = tmp_path / "mlruns"
    cfg.paths.mlflow_db = tmp_path / "mlflow.db"
    cfg.paths.tensorboard = tmp_path / "tb"
    cfg.paths.logs = tmp_path / "logs"
    cfg.policy.max_troiani_gpus = 6
    cfg.policy.max_jobs = 4
    cfg.policy.cooldown_s = 0
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def service(tmp_cfg: PlatformConfig) -> PlatformService:
    return PlatformService(tmp_cfg, store=StateStore(tmp_cfg.paths.store))


def make_spec(**kwargs) -> JobSpec:
    base = dict(
        name="job-a",
        command=("python", "-m", "troiani_platform.training.dummy_train"),
        gpus=1,
        min_gpu_memory_gb=40,
        priority=Priority.OPPORTUNISTIC,
        preemptible=True,
    )
    base.update(kwargs)
    spec = JobSpec(**base)
    spec.validate()
    return spec
