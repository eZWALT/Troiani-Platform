from troiani_platform.worker.watchdog import Watchdog


def test_control_lost_after_grace():
    dog = Watchdog(heartbeat_s=1, stale_s=5, control_lost_grace_s=10)
    dog.mark_control(True)
    assert not dog.control_lost(now=dog.last_control_ok + 3)
    assert dog.control_lost(now=dog.last_control_ok + 11)
