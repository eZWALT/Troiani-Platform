# Autocritical backlog

Priority: P0 now, P1 soon, P2 later. Agents: grab a P0/P1, mark it in IMPROVEMENT_LOG.md, implement, test, deploy Atlas/Uranus.

## P0 — operator UI

- [x] Horizontal tabbed dashboard (IBM Plex / lab mono)
- [x] Policy editor: named windows with days, add/remove, presets
- [x] Launch + resume + preempt + cancel from UI
- [x] Drain/release button per GPU (Troiani only; researcher disabled)
- [x] Force-kill Troiani job (after confirm; never foreign PIDs)
- [x] Run table: duration, tokens, tok/s, step, loss, node, GPUs
- [x] Live GPU util/memory charts (no full-page reload)
- [x] Worker heartbeat / stale worker visible
- [x] Job log tail in UI (`GET /v1/jobs/{id}/log`)

## P0 — statistics

- [x] Persist tokens_ingested + tokens_per_sec on Job/Run
- [x] Runtime from started_at → now/ended_at
- [x] Cluster cards: Troiani GPU-hours, researcher GPU-hours, tokens total
- [x] Dummy trainer reports tokens so smokes fill the UI

## P1 — multi-node / smokes

- [x] Worker reports checkpoint manifests to control
- [x] Preferred node placement
- [x] Live 1-GPU smoke on idle Atlas GPU1 (`job-85a8776fbf7b` COMPLETED after preempt/resume)
- [x] Live 2-GPU smoke on idle Uranus 0+1 (`job-b03c52ad16ae` COMPLETED)
- [x] Live 2-node smoke: Atlas idle + Uranus idle (two jobs, not NCCL)
- [x] Placement still refuses 70GB jobs on 40GB cards
- [x] Stale worker → do not leave jobs RUNNING forever

## P1 — monitoring honesty

- [x] Intent uniqueness + 6h freshness (stop score=100 spam)
- [x] Annoyance score per node
- [x] Per-user GPU occupancy time
- [x] Overnight window presets on the cluster as disabled rows (all-disabled = always on)

## P0 — policy composition (2026-09-10)

- [x] Named rules: weeknights, lunch, weekends, weekend-nights, special, festive
- [x] Current vs proposed before save
- [x] STOP ALL latch + Resume (Troiani only)
- [x] Aggressiveness low/mid/extreme mapped to memory_busy_gb / cooldown / caps
- [x] SM vs VRAM honesty on cards, table, chart

## P1 — throughput bench matrix (not M16)

- [x] Fit estimator + unit tests (2B×batch16 fails 40GB; 350M×batch1 fits 40 and 80)
- [x] User token-budget guidance in `.agents/SCALING.md` + MEMORY pointer
- [x] Live tokens/s on idle Atlas gpu1 (40GB) and Uranus gpu0 (80GB): 350M / 1B / 2B×1; 350M×16 OOM on 40GB (no fake tok/s)
- [x] Table in `.agents/THROUGHPUT.md`
- [ ] Leftover cells: Uranus 777M, 1B×16 / 2B×8–16 on 80GB (stopped when csp appeared on Uranus gpu0)
- M16 multi-day Troiani 1B pretrain is **still later**

## P1 — ingest / infra

- [x] SkyPilot-shaped YAML + script entrypoint (`/v1/jobs/from-yaml`)
- [x] Named checkpoints `{experiment}-{kind}-{step}`
- [x] Infra tab: disk, top, checkpoint GC
- [x] Activity per-user view
- [x] Async process checkpoint writer (regular off-trainer; preempt/shutdown sync)

## P2

- [x] Evaluation hook actually scheduled
- [x] Dataset manifest fingerprint in the run pane (API exists)
- [x] Dark theme toggle
- [x] Proxy remote worker logs beyond heartbeat tail

## Never

- Kill researcher processes
- Touch Troiani LLM repo
- Fake GPU availability
- Multi-node NCCL orchestration in v1
