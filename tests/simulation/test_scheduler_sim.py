from troiani_platform.config import PolicyConfig
from troiani_platform.models import Job, JobState
from troiani_platform.scheduler.scheduler import CooperativeScheduler
from troiani_platform.simulation.cluster import SimulatedCluster
from tests.conftest import make_spec


def test_six_gpu_researcher_arrives():
    cluster = SimulatedCluster()
    job_a = Job(
        id="job-a",
        spec=make_spec(name="job-a", gpus=2),
        state=JobState.RUNNING,
        assigned_gpus=["gpu-atlas-0", "gpu-atlas-1"],
        assigned_node="atlas",
    )
    job_b = Job(id="job-b", spec=make_spec(name="job-b", gpus=2), state=JobState.PENDING)
    cluster.researcher_claim("gpu-atlas-0", username="gkoutr")
    cluster.researcher_claim("gpu-atlas-1", username="gkoutr")
    sched = CooperativeScheduler(PolicyConfig(max_troiani_gpus=6, cooldown_s=0))
    decisions = sched.schedule([job_a, job_b], cluster.gpus)
    preempts = [d for d in decisions if d.kind == "PREEMPT"]
    allocs = [d for d in decisions if d.kind == "ALLOCATE"]
    assert preempts and preempts[0].job_id == "job-a"
    assert allocs and allocs[0].job_id == "job-b"
    assert allocs[0].placement is not None
    assert allocs[0].placement.node == "uranus"
