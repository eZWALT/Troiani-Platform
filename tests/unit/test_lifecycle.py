from troiani_platform.worker.lifecycle import request_checkpoint, terminate
from troiani_platform.worker.worker import Worker


def test_signals_missing_pid_are_harmless():
    assert request_checkpoint(9_999_999) is False
    assert terminate(9_999_999) is False


def test_preempt_dead_proc_does_not_crash(tmp_cfg):
    worker = Worker(tmp_cfg, node="atlas", worker_id="atlas-test")

    class Dead:
        pid = 9_999_999

        def poll(self):
            return 0

    worker.procs["job-dead"] = Dead()
    worker.apply_actions([{"kind": "PREEMPT", "job_id": "job-dead"}])
    worker.apply_actions([{"kind": "STOP", "job_id": "job-dead"}])
    worker.apply_actions([{"kind": "KILL", "job_id": "job-dead"}])
