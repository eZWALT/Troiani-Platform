import pytest

from troiani_platform.errors import CheckpointError
from troiani_platform.models import CheckpointKind
from troiani_platform.simulation.faults import FaultInjector
from troiani_platform.training.checkpoint import CheckpointManager, SaveRequest


def test_disk_full_fault(tmp_path):
    mgr = CheckpointManager(tmp_path)
    FaultInjector(mgr).enable("disk_full")
    with pytest.raises(CheckpointError):
        mgr.save(
            SaveRequest(
                run_id="r",
                job_id="j",
                step=1,
                kind=CheckpointKind.REGULAR,
                state={"step": 1},
            )
        )
