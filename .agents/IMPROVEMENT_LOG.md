# Improvement log

## 2026-09-10 17:32

- Created `.agents` memory.

## 2026-09-10 17:40

- Live dashboard (`/static/app.js`) with 2s poll, util chart, policy form, drain/kill, tokens + runtime.
- APIs: `PUT /v1/policy`, `POST /v1/gpus/{uuid}/drain|undrain`, `POST /v1/jobs/{id}/kill`.
- Dummy trainer reports tokens. 38 tests green.
- Next loop: multi-node checkpoint metadata, 2-GPU smoke, eval hooks, annoyance score.

## 2026-09-10 17:45

- Per-node sharing / annoyance score on `GET /v1/status` as `annoyance: [{node, score, reasons}]`.
- Formula (deterministic): `min(100, round(12*LOGIN + 20*FOREIGN_PROCESS + 50*max(live RESEARCHER frac, occupancy-time frac)))`.
- Dashboard "Sharing" table next to workers. Policy / drain / tokens UI unchanged.
- Researcher processes are never killed; this is an operator courtesy score only.

## 2026-09-10 17:50

- Stale worker heartbeat (`policy.worker_stale_s`): control plane reclaims Troiani jobs that would otherwise sit RUNNING and keep GPUs.
- Valid path only: `… → PREEMPTED → QUEUED`. GPU assignment released. `WORKER_STALE` emitted once.
- No remote PID kills (researcher or otherwise). Worker watchdog `control_lost` release left intact.
- Reclaim runs on `tick()` and `status()` so dashboard polls recover even with no heartbeats. Stale-node GPUs are not rescheduled until that worker returns.

## 2026-09-10 17:52

- Evaluation hooks actually run: `JobSpec.evaluation.every_steps` + argv `command` are exported by the runner (`EVAL_EVERY_STEPS` / `EVAL_COMMAND`). Dummy trainer runs the command every N steps via `subprocess.run(..., shell=False)` and POSTs `val_loss` to `/v1/internal/metrics`.
- Placement tests tightened: 2-GPU same-node, A100 40 vs 80 preference, 70GB refused on 40GB-only / researcher-claimed 80GB pools. `config/jobs/smoke-2gpu.yaml` used in-process only (no live cluster submit).
- `pytest -q`: 77 passed.

## 2026-09-10 17:55

- Multi-node checkpoint metadata: `POST /v1/internal/checkpoint` records a `CheckpointRecord` from a worker-posted manifest (`id`, `run_id`, `job_id`, `step`, `kind`, `healthy`, `checksum`, `path`).
- Dummy trainer POSTs after each successful save; worker also reports latest local checkpoint on heartbeat/reap.
- `job_exit` / `status` keep `last_checkpoint_id` even when files are not on Atlas disk.
- `pytest -q`: 81 passed.

## 2026-09-10 18:10

- Named policy windows: `{name,start,end,days,enabled}`. Overnight belongs to the start weekday. Empty list still means always-on opportunistic.
- Dashboard rewritten: IBM Plex, horizontal tabs (Cluster / Jobs / Launch / Policy / Runs / Activity), GPU strip, launch form, window rows, lineage/reproduce inspect, job log tail.
- `preferred_node` on JobSpec; templates at `GET /v1/templates`; logs via worker heartbeat → `GET /v1/jobs/{id}/log`.
- CLI: `job log`, `job templates`. `TROIANI_PLATFORM_URL` overrides control public URL (laptop tunnel).
- Annoyance: unique users/GPUs + 6h freshness so leftover LOGIN spam cannot pin the score at 100.
- Persistent context: `.agents/SPEC.md` is the 40-task checklist. 15m loop already running; add visible 12h countdown.
- `pytest -q`: 87 passed.

## 2026-09-10 18:05

- Live smokes on idle cards only. `gkoutr` 642620 and `csp` vLLM 2524772/2524773 never touched.
- `job-b03c52ad16ae` 2-GPU Uranus COMPLETED (step 80, ckpt-000080, 327680 tokens).
- Atlas 1-GPU `job-85a8776fbf7b` preempted at step 48; dummy finished 50 during SIGUSR1; worker died on ProcessLookupError. Fixed: missing PID is harmless, worker loop catches exceptions.
- Max-runtime was wall-clock (blocked resume after preemption). Now useful GPU time via `trained_s` + `active_since`.
- Resumed atlas job COMPLETED from `ckpt-000050`. Concurrent Uranus 1-GPU `job-dbf711dea076` COMPLETED. Log tail from Uranus visible on Atlas control.
- `pytest -q`: 90 passed.
- 12h visible countdown started. 15m `AGENT_LOOP_TICK_platform12h` still armed.

## 2026-09-10 18:10 loop tick

- Per-user GPU occupancy: `InfraMetrics` accumulates seconds per `(node, username)` split into troiani/other. Exposed on `GET /v1/status` as `user_occupancy`.
- Cluster tab: "Who has the GPUs" table. Runs tab shows dataset name@revision.
- Did not enable overnight windows on the live cluster (operator decision; empty = any hour, still yield).
- `pytest -q`: 90 passed. Deploy + restart control only.

## 2026-09-10 18:15 loop tick 2

- Heartbeat now reconciles dead trainers: `alive=false` on RUNNING/CHECKPOINTING calls `job_exit` instead of sitting there until worker-stale (45s).
- A run that already printed `completed step=` is COMPLETED even if a preempt raced the last step.
- Unfinished preempt still requeues. `pytest -q`: 93 passed. Deploy + restart control only.
- Overnight windows stay off on the live cluster (operator decision).

## 2026-09-10 18:25 loop tick 3

- Infra metrics (GPU-hours, per-user occupancy, util history) persist in SQLite `infra_metrics` so a control restart does not wipe the sharing table.
- Dark/Light toggle (IBM Plex, localStorage `tp-theme`) in the header.
- `pytest -q`: 94 passed. Deploy + restart control only.

## 2026-09-10 18:40 loop tick 4

- Workers ship incremental log chunks (32KB, byte offsets). Control appends up to 256KB in SQLite so Uranus logs survive on the Atlas UI beyond one heartbeat tail.
- Failed heartbeats do not advance the offset (no silent gaps). Reap flushes leftover bytes.
- `pytest -q`: 95 passed. Deploy; restart control + workers. Overnight windows still off.

## 2026-09-10 18:55 loop tick 5

- All-disabled (or empty) windows mean always-on opportunistic. Enabling any row restores time gating.
- Weeknights + lunch presets now exist as disabled rows in config and in the Policy tab. Flip `on` only when you want them.
- `pytest -q`: 96 passed. Deploy + restart control only.

## 2026-09-10 19:20

- SM vs VRAM honesty: discovery reads `gpu_util` + `fb_memory_usage` + `memory_util`. Status exposes `sm`/`vram_*`/`mem_ctrl`/`ref`. Cards, table, chart show both.
- Policy is composed rules (weeknights, lunch, weekends, weekend-nights, special, festive). Current vs proposed. STOP ALL / Resume. Aggressiveness low/mid/extreme maps onto `memory_busy_gb`, cooldown, caps.
- SkyPilot-shaped YAML ingest (`run`/`entrypoint` → argv via shlex). Named checkpoints `{exp}-{kind}-{step}`. Process async writer for regular saves. Infra tab + per-user activity.
- Visual pass: readable policy summaries, On/Off partial stop, TB disk units. Live STOP ALL tested then resumed. Windows stay disabled.
- `pytest -q`: 107 passed. Deploy Atlas+Uranus. Push to GitHub.
