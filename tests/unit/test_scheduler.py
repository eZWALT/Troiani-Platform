from troiani_platform.config import PolicyConfig
from troiani_platform.models import Job, JobState
from troiani_platform.scheduler.scheduler import CooperativeScheduler
from troiani_platform.simulation.cluster import SimulatedCluster
from tests.conftest import make_spec


def test_researcher_preempts_troiani():
    cluster = SimulatedCluster()
    job = Job(id="t1", spec=make_spec(gpus=2), state=JobState.RUNNING, assigned_gpus=["gpu-atlas-0", "gpu-atlas-1"])
    cluster.gpus[0] = cluster.gpus[0].__class__.from_dict({**cluster.gpus[0].to_dict(), "job_id": "t1"})
    cluster.gpus[1] = cluster.gpus[1].__class__.from_dict({**cluster.gpus[1].to_dict(), "job_id": "t1"})
    cluster.researcher_claim("gpu-atlas-0")
    sched = CooperativeScheduler(PolicyConfig(max_troiani_gpus=6, cooldown_s=0))
    decisions = sched.schedule([job], cluster.gpus)
    kinds = {d.kind for d in decisions}
    assert "PREEMPT" in kinds


def test_fairness_cap():
    cluster = SimulatedCluster()
    running = Job(id="t1", spec=make_spec(gpus=2), state=JobState.RUNNING, assigned_gpus=["gpu-uranus-0", "gpu-uranus-1"])
    waiting = Job(id="t2", spec=make_spec(name="job-b", gpus=4), state=JobState.PENDING)
    sched = CooperativeScheduler(PolicyConfig(max_troiani_gpus=4, max_gpus_per_job=2, cooldown_s=0))
    decisions = sched.schedule([running, waiting], cluster.gpus)
    holds = [d for d in decisions if d.kind == "HOLD" and d.job_id == "t2"]
    assert holds
