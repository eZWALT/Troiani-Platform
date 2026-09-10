from pathlib import Path

import yaml

from troiani_platform.models import Job, JobSpec
from troiani_platform.scheduler.placement import place_job
from troiani_platform.simulation.cluster import SimulatedCluster, default_pool
from tests.conftest import make_spec

_REPO = Path(__file__).resolve().parents[2]


def _job_yaml(name: str) -> JobSpec:
    raw = yaml.safe_load((_REPO / "config" / "jobs" / name).read_text())
    return JobSpec.from_dict(raw)


def test_70gb_refuses_40gb_cards():
    job = Job(id="wide", spec=make_spec(min_gpu_memory_gb=70, gpus=1))
    placement = place_job(job, default_pool())
    assert placement is not None
    assert placement.node == "uranus"
    assert all(u.startswith("gpu-uranus") for u in placement.gpu_uuids)


def test_70gb_refused_when_only_40gb_available():
    atlas = [gpu for gpu in default_pool() if gpu.memory_gb < 70]
    assert atlas and all(gpu.memory_gb == 40 for gpu in atlas)
    job = Job(id="too-fat", spec=make_spec(min_gpu_memory_gb=70, gpus=1))
    assert place_job(job, atlas) is None


def test_70gb_refused_when_80gb_cards_are_researcher_claimed():
    cluster = SimulatedCluster()
    for uuid in ("gpu-uranus-0", "gpu-uranus-1", "gpu-uranus-2", "gpu-uranus-3"):
        cluster.researcher_claim(uuid)
    job = Job(id="wide", spec=make_spec(min_gpu_memory_gb=70, gpus=1))
    assert place_job(job, cluster.gpus) is None


def test_two_gpu_stays_on_one_node():
    job = Job(id="pair", spec=make_spec(gpus=2, min_gpu_memory_gb=40))
    placement = place_job(job, default_pool())
    assert placement is not None
    assert len(placement.gpu_uuids) == 2
    assert len({placement.node}) == 1


def test_two_gpu_does_not_split_across_nodes():
    cluster = SimulatedCluster()
    cluster.researcher_claim("gpu-atlas-0")
    job = Job(id="pair", spec=make_spec(gpus=2, min_gpu_memory_gb=40))
    placement = place_job(job, cluster.gpus)
    assert placement is not None
    assert placement.node == "uranus"
    assert len(placement.gpu_uuids) == 2


def test_preferred_40gb_selects_atlas_a100():
    job = Job(id="small", spec=make_spec(gpus=1, min_gpu_memory_gb=40, preferred_gpu_type="40GB"))
    placement = place_job(job, default_pool())
    assert placement is not None
    assert placement.node == "atlas"
    assert placement.gpu_uuids[0].startswith("gpu-atlas")


def test_preferred_80gb_selects_uranus_a100():
    job = Job(id="fat", spec=make_spec(gpus=1, min_gpu_memory_gb=40, preferred_gpu_type="80GB"))
    placement = place_job(job, default_pool())
    assert placement is not None
    assert placement.node == "uranus"
    assert all(u.startswith("gpu-uranus") for u in placement.gpu_uuids)


def test_two_gpu_80gb_stays_on_uranus():
    job = Job(id="pair80", spec=make_spec(gpus=2, min_gpu_memory_gb=70, preferred_gpu_type="80GB"))
    placement = place_job(job, default_pool())
    assert placement is not None
    assert placement.node == "uranus"
    assert len(placement.gpu_uuids) == 2


def test_preferred_node_is_honored():
    job = Job(id="atlas-only", spec=make_spec(gpus=1, preferred_node="atlas"))
    placement = place_job(job, default_pool())
    assert placement is not None
    assert placement.node == "atlas"


def test_preferred_node_refuses_other_nodes():
    uranus = [gpu for gpu in default_pool() if gpu.node == "uranus"]
    job = Job(id="atlas-only", spec=make_spec(gpus=1, preferred_node="atlas"))
    assert place_job(job, uranus) is None


def test_smoke_2gpu_yaml_places_two_a100s_on_one_node():
    spec = _job_yaml("smoke-2gpu.yaml")
    assert spec.gpus == 2
    assert spec.min_gpu_memory_gb == 40
    assert spec.preferred_gpu_type == "A100"
    job = Job(id="smoke-2gpu", spec=spec)
    placement = place_job(job, default_pool())
    assert placement is not None
    assert len(placement.gpu_uuids) == 2
    assert len({placement.node}) == 1
    pool = {gpu.uuid: gpu for gpu in default_pool()}
    for uuid in placement.gpu_uuids:
        gpu = pool[uuid]
        assert gpu.model == "A100"
        assert gpu.memory_gb >= 40
        assert gpu.node == placement.node
