from troiani_platform.models import AnomalyKind
from troiani_platform.training.health import HealthMonitor


def test_nan_and_inf():
    mon = HealthMonitor()
    assert AnomalyKind.LOSS_NAN in mon.observe(ts=1, loss=float("nan"), step=1)
    assert AnomalyKind.LOSS_INF in mon.observe(ts=2, loss=float("inf"), step=2)


def test_spike_and_stall():
    mon = HealthMonitor(stall_seconds=10, loss_spike_factor=4)
    mon.observe(ts=0, loss=1.0, step=1)
    assert AnomalyKind.LOSS_SPIKE in mon.observe(ts=1, loss=9.0, step=2)
    mon.observe(ts=10, loss=1.0, step=3)
    assert AnomalyKind.TRAINING_STALLED in mon.observe(ts=25, loss=1.0)


def test_throughput_collapse():
    mon = HealthMonitor(throughput_collapse_factor=0.5)
    mon.observe(ts=0, step=1, throughput=100)
    assert AnomalyKind.THROUGHPUT_COLLAPSE in mon.observe(ts=1, step=2, throughput=20)
