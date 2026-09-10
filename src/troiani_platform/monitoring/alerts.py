from __future__ import annotations

from troiani_platform.log import get_logger
from troiani_platform.models import AnomalyKind, Event, IntentSignal

log = get_logger("troiani.platform.alerts")


def notify_anomaly(job_id: str, kinds: list[AnomalyKind]) -> Event:
    log.warning("anomaly", extra={"job_id": job_id, "event": ",".join(k.value for k in kinds)})
    return Event(type="ANOMALY_DETECTED", job_id=job_id, payload={"kinds": [k.value for k in kinds]})


def notify_intent(signal: IntentSignal) -> Event:
    log.info(
        "intent",
        extra={"event": signal.kind.value, "node": signal.node, "gpu": signal.gpu_uuid or ""},
    )
    return Event(
        type="INTENT_SIGNAL",
        payload=signal.to_dict(),
        source="discovery",
    )
