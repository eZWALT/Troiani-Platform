# Troiani Platform — persistent spec (the 40-task prompt)

This is the long-lived agent context. Read MEMORY.md + BACKLOG.md + this file at every loop tick.

Repo: `https://github.com/eZWALT/Troiani-Platform`
Laptop workspace: `~/Escritorio/SIDE/Troiani-Platform`
Runtime: Atlas `~/Troiani-Platform` (control+worker), Uranus `~/Troiani-Platform` (worker).
Do **not** write into the Troiani LLM repo.

Hardware: Atlas 2× A100 40GB (GPU0 often `gkoutr`). Uranus 4× A100 80GB (GPU2–3 often `csp` vLLM). Never kill researcher PIDs.

## Original milestone checklist

- [x] M1 Discovery / reuse existing boring Python + SQLite + FastAPI
- [x] M2 GPU discovery (UUID, VRAM, util, processes, node)
- [x] M3 Job spec + state machine
- [x] M4 Scheduler simulation
- [x] M5 Single-node worker
- [x] M6 Checkpoint manager (atomic, manifest, validate, retention)
- [x] M7 Preempt → checkpoint → release → resume
- [x] M8 Run / artifact / git / env tracking
- [x] M9 Training health (NaN/Inf/stall/spike)
- [x] M10 Latest-valid fallback + conservative rollback
- [x] M11 Heterogeneous placement (40GB ≠ 80GB, 1/2/4 GPU, preferred node)
- [x] M12 Control plane API + persistent SQLite
- [x] M13 Dashboard (tabs, launch, policy rows, GPU util)
- [x] M14 Named policy rules + STOP ALL + aggressiveness + live policy PUT
- [x] M15 Watchdog + stale worker + fault injection hooks
- [ ] M16 Multi-day real Troiani model (not dummy) — later, idle GPUs only
- Token budget for a 1B is in `.agents/SCALING.md` (20× is compute-optimal, not good enough). Throughput probe is **not** M16: `.agents/THROUGHPUT.md`.

## Extra ROI the user added

- [x] Dataset fingerprint (manifest hash, not DVC)
- [x] Eval hook argv every N steps → MLflow val_loss
- [x] Checkpoint GC (last 5 / best 3 / milestones / latest preemption)
- [x] MLflow + TensorBoard local persistent
- [x] Annoyance / other-user signals (unique + 6h, not accumulating spam)
- [x] UI launch / resume / preempt / release-all
- [x] Multiple named policy rules (days + hours) + weekends + festive/special chips
- [x] STOP ALL emergency latch (Troiani preempt only)
- [x] SM + VRAM both shown (never one "util" lie)
- [x] SkyPilot-ish YAML ingest
- [x] Infra / top / checkpoint GC tab
- [x] Per-user activity
- [x] Async process checkpoint writer
- [x] Job log tail via worker heartbeat
- [x] Real 1-GPU / 2-GPU / 2-node smokes on idle cards (do every deploy)
- [ ] Intent-score decay further if still noisy
- [ ] Grafana/Prometheus — skip unless MLflow+TB fail
- [x] Token-budget guidance (SCALING.md) + 1-GPU throughput estimator (THROUGHPUT.md). Not M16.

## Hard rules

- Opportunistic and preemptible. Researchers always have priority.
- No disguise, no quota bypass, no killing researcher processes.
- Command is a YAML/JSON argv list, never a shell string.
- Placement is same-node. "2 nodes" means one job on Atlas idle + one on Uranus idle. Not NCCL.
- 70GB must not fit 40GB.
- Deploy with rsync excluding `.venv`, `var/`, `.git`. Restart only control/workers.

## How to run

```bash
ssh -N -L 8787:127.0.0.1:8787 -L 5000:127.0.0.1:5000 -L 6006:127.0.0.1:6006 atlas
# http://127.0.0.1:8787
TROIANI_PLATFORM_URL=http://127.0.0.1:8787 troiani-platform status
```

## Loop contract

A 15-minute `AGENT_LOOP_TICK_platform12h` process plus a visible 12h countdown.
On each tick: read these files, pick the highest unfinished P0/P1, implement, pytest, deploy, smoke idle GPUs only, update IMPROVEMENT_LOG.md.
Even if "done", keep polishing UI, policy, observability, and tests.
