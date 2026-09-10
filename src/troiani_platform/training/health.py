from __future__ import annotations

import math
from dataclasses import dataclass, field

from troiani_platform.models import AnomalyKind


@dataclass
class HealthMonitor:
    stall_seconds: float = 600
    loss_spike_factor: float = 8.0
    throughput_collapse_factor: float = 0.5
    underutil_threshold: float = 5.0
    previous_loss: float | None = None
    last_progress_ts: float | None = None
    baseline_throughput: float | None = None
    last_step: int | None = None
    history: list[AnomalyKind] = field(default_factory=list)

    def observe(
        self,
        *,
        ts: float,
        loss: float | None = None,
        step: int | None = None,
        throughput: float | None = None,
        gpu_util: float | None = None,
        dataloader_latency_ms: float | None = None,
        checkpoint_failed: bool = False,
    ) -> list[AnomalyKind]:
        found: list[AnomalyKind] = []
        if loss is not None:
            if math.isnan(loss):
                found.append(AnomalyKind.LOSS_NAN)
            elif math.isinf(loss):
                found.append(AnomalyKind.LOSS_INF)
            elif self.previous_loss and self.previous_loss > 0 and loss > self.previous_loss * self.loss_spike_factor:
                found.append(AnomalyKind.LOSS_SPIKE)
            if not found:
                self.previous_loss = loss
        if step is not None and (self.last_step is None or step > self.last_step):
            self.last_step = step
            self.last_progress_ts = ts
        if self.last_progress_ts is not None and (ts - self.last_progress_ts) >= self.stall_seconds:
            found.append(AnomalyKind.TRAINING_STALLED)
        if throughput is not None:
            if self.baseline_throughput is None and throughput > 0:
                self.baseline_throughput = throughput
            elif (
                self.baseline_throughput
                and throughput < self.baseline_throughput * self.throughput_collapse_factor
            ):
                found.append(AnomalyKind.THROUGHPUT_COLLAPSE)
        if gpu_util is not None and gpu_util < self.underutil_threshold:
            found.append(AnomalyKind.GPU_UNDERUTILIZATION)
        if dataloader_latency_ms is not None and dataloader_latency_ms > 5000:
            found.append(AnomalyKind.DATA_PIPELINE_STARVATION)
        if checkpoint_failed:
            found.append(AnomalyKind.CHECKPOINT_FAILURE)
        self.history.extend(found)
        return found
