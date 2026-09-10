from troiani_platform.jobs.skypilot import spec_from_task_yaml
from troiani_platform.models import JobSpec


def test_skypilot_run_string_becomes_argv():
    spec = spec_from_task_yaml(
        {
            "name": "bench",
            "resources": {"accelerators": "A100-80GB:2"},
            "run": "python -m troiani_platform.training.dummy_train --steps 10",
            "workdir": "/tmp/work",
        }
    )
    assert spec["gpus"] == 2
    assert spec["min_gpu_memory_gb"] == 80.0
    assert spec["preferred_gpu_type"] == "A100-80GB"
    assert spec["command"][:3] == ["python", "-m", "troiani_platform.training.dummy_train"]
    JobSpec.from_dict(spec)


def test_entrypoint_and_args():
    spec = spec_from_task_yaml({"name": "e1", "entrypoint": "train.py", "args": ["--epochs", "2"]})
    assert spec["command"] == ["train.py", "--epochs", "2"]
