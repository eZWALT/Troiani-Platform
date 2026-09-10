from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from troiani_platform.models import GPUResource, IntentKind, IntentSignal, Occupancy, parse_ts

# Deterministic per-node sharing / annoyance score (operator courtesy only).
#
# Troiani never kills researcher processes. This number answers: "are we
# sharing this box with other people?" It is computed only from stored
# IntentSignals and GPU occupancy — no extra probes, no process signals.
#
# Per node (unique + recent, so heartbeats cannot inflate the score):
#   login_n   = unique usernames with LOGIN in the last 6h
#   foreign_n = unique (username, gpu) FOREIGN_PROCESS in the last 6h
#   live_frac = (# cards currently Occupancy.RESEARCHER) / (# cards)
#               0 if the node has no GPUs in the snapshot
#   time_frac = researcher_gpu_seconds / observed_gpu_seconds
#               0 if occupancy time has not been accumulated yet
#   researcher_frac = max(live_frac, time_frac)
#
#   raw   = 12 * login_n + 20 * foreign_n + 50 * researcher_frac
#   score = min(100, int(raw + 0.5))          # half-up, inclusive 0–100
#
# Same inputs always yield the same list, sorted by node name. A reason
# string is emitted only when that term is greater than zero.

LOGIN_WEIGHT = 12
FOREIGN_WEIGHT = 20
OCCUPANCY_WEIGHT = 50
INTENT_MAX_AGE_S = 6 * 3600


def _fresh_intents(
    intents: Iterable[IntentSignal],
    now: datetime | None = None,
    max_age_s: float = INTENT_MAX_AGE_S,
) -> list[IntentSignal]:
    now_dt = now or datetime.now(timezone.utc)
    out: list[IntentSignal] = []
    for signal in intents:
        ts = parse_ts(signal.ts)
        if ts is None or (now_dt - ts).total_seconds() <= max_age_s:
            out.append(signal)
    return out


def compute_annoyance(
    intents: Iterable[IntentSignal],
    gpus: Iterable[GPUResource],
    occupancy_seconds: Mapping[str, Mapping[str, float]] | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    occ = occupancy_seconds or {}
    by_node_intents: dict[str, list[IntentSignal]] = defaultdict(list)
    for signal in _fresh_intents(intents, now=now):
        node = (signal.node or "").strip() or "unknown"
        by_node_intents[node].append(signal)

    by_node_gpus: dict[str, list[GPUResource]] = defaultdict(list)
    for gpu in gpus:
        node = (gpu.node or "").strip() or "unknown"
        by_node_gpus[node].append(gpu)

    nodes = sorted(set(by_node_intents) | set(by_node_gpus) | set(occ))
    out: list[dict[str, Any]] = []
    for node in nodes:
        signals = by_node_intents.get(node, [])
        cards = by_node_gpus.get(node, [])
        login_users = sorted(
            {
                (s.username or "").strip()
                for s in signals
                if s.kind == IntentKind.LOGIN and (s.username or "").strip()
            }
        )
        login_n = len(login_users)
        foreign_n = len(
            {
                ((s.username or "").strip(), s.gpu_uuid or "")
                for s in signals
                if s.kind == IntentKind.FOREIGN_PROCESS
            }
        )
        live_researcher = sum(1 for g in cards if g.occupancy == Occupancy.RESEARCHER)
        n_cards = len(cards)
        live_frac = (live_researcher / n_cards) if n_cards else 0.0

        bucket = occ.get(node) or {}
        researcher_s = float(bucket.get("researcher") or 0.0)
        observed_s = float(bucket.get("total") or 0.0)
        time_frac = (researcher_s / observed_s) if observed_s > 0 else 0.0
        researcher_frac = max(live_frac, time_frac)

        raw = LOGIN_WEIGHT * login_n + FOREIGN_WEIGHT * foreign_n + OCCUPANCY_WEIGHT * researcher_frac
        score = min(100, int(raw + 0.5))

        reasons: list[str] = []
        if login_n:
            who = f" ({', '.join(login_users)})" if login_users else ""
            reasons.append(f"{login_n} LOGIN{who}")
        if foreign_n:
            reasons.append(f"{foreign_n} FOREIGN_PROCESS")
        if live_researcher:
            reasons.append(f"{live_researcher}/{n_cards} GPUs RESEARCHER now")
        if observed_s > 0 and researcher_s > 0:
            reasons.append(
                f"researcher occupancy {time_frac:.2f} ({researcher_s:.0f}s / {observed_s:.0f}s)"
            )

        out.append({"node": node, "score": score, "reasons": reasons})
    return out
