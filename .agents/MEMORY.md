# Troiani-Platform agent memory

Updated: 2026-09-10 19:40 Europe/Madrid

## What this is

Cooperative preemptible training platform. Researchers always win.
Repo: `~/Escritorio/SIDE/Troiani-Platform` (Cursor workspace)
Remote: `git@github.com:eZWALT/Troiani-Platform.git`
Runtime: Atlas `~/Troiani-Platform` (control + worker) and Uranus `~/Troiani-Platform` (worker)
Do **not** write into the Troiani LLM repo.

## Hardware (honest)

- Atlas `192.168.1.17`: 2× A100 40GB. GPU0 often `gkoutr` (RESEARCHER, high VRAM / low SM). GPU1 often idle.
- Uranus `192.168.1.18`: 4× A100 80GB. GPU2–3 often `csp` vLLM (RESEARCHER, high SM + VRAM). GPU0–1 often idle.
- Never kill, hide, or interfere with researcher processes.
- "Kill" / Drain / STOP ALL means Troiani work only. Researcher cards stay Blocked.

## Live services (Atlas)

- Control + dashboard: `0.0.0.0:8787`
- Tunnel: `ssh -N -L 8787:127.0.0.1:8787 -L 5000:127.0.0.1:5000 -L 6006:127.0.0.1:6006 atlas`
- http://127.0.0.1:8787 or LAN http://192.168.1.17:8787
- CLI: `TROIANI_PLATFORM_URL=http://127.0.0.1:8787 troiani-platform …`
- Deploy: `scripts/deploy.sh` (excludes `.venv` and `var/`)

## Architecture

- Control + SQLite + one worker per node. Package `troiani_platform`.
- New modules: `infra/` (notation, storage), `jobs/skypilot.py`, `training/async_writer.py`
- GPU JSON always has `sm` / `sm_util` (compute %) and `vram_*` (framebuffer). `utilization` is SM, never VRAM.
- GPU refs: `atlas/gpu0`. Checkpoints: `{experiment}-{kind}-{step:06d}` plus `experiments/{exp}/{run_id}` index.
- Policy is composed named rules + `stop_all` latch + `aggressiveness` low/mid/extreme.
- Empty / all-disabled rules = always-on opportunistic unless STOP ALL.
- SkyPilot-shaped YAML: `POST /v1/jobs/from-yaml` (shlex argv, never shell).
- Infra: `GET /v1/infra`, `POST /v1/infra/gc`. Activity: `activity_users` + per-user tab.

## UI

Tabs: Cluster / Jobs / Launch / Policy / Runs / Activity / Infra.
Subtitle: `2 nodes · A100`.
Policy: emergency STOP ALL / Resume, aggressiveness buttons, rule chips, current vs proposed, On/Off partial stop.
Cluster GPU cards show SM and VRAM separately.

## Pretraining tokens (1B dense) — do not sloganize

Chinchilla ~20 tok/param is **compute-optimal**, not “good enough”.
10B undertrained · 20B Chinchilla-compute · 50B solid · 100B quite good · 300B very strong · 1T seriously overtrained / potentially excellent · 3T TinyLlama-extreme · 5–10T diminishing.
Interesting regime **100B–1T**. For a 1B meant for cheap inference today: target **≈300B–1T high-quality tokens**, not blindly 20B.
Once in the hundreds of billions, **data quality > token counter**. Prefer 1B × 500B excellent tokens over 1B × 3T garbage.
Full table: `.agents/SCALING.md`. Throughput / VRAM fit: `.agents/THROUGHPUT.md`.

## Throughput probe (not M16)

- Module: `python -m troiani_platform.training.throughput --params 1B --batch 4 --seq 2048 --steps 30`
- Estimator first (FIT / NO-FIT on 40 vs 80). Seq default **2048**. Same-node, 1 GPU, not NCCL.
- Live timed steps only on idle cards. GPU0 Atlas often gkoutr; Uranus GPU2–3 often csp vLLM.
- Live (2026-09-10): Atlas gpu1 350M×1 = 47.5k tok/s; 1B×1 = 22.7k; 2B×1 = 12.6k. Uranus gpu0 350M×1 = 50.5k; 1B×1 = 23.9k; 2B×1 = 13.2k. 350M×16 OOM on 40GB (39.7 GiB on 80GB). 2B×8/16 NO-FIT 40GB.
- ETA (1 GPU, probe rate): 1B×1 Atlas 100B ≈ **51 d always-on** / **142 d weeknights** (35.8%). Live windows are all-off = 100% duty. Hypothetical only; windows were not enabled. Full table in THROUGHPUT.md.
- Do not invent tokens/s. M16 (real 1B pretrain) is still later.

## Invariants

- No researcher PID kill
- Do not enable overnight/weekend windows live unless the user turns them on
- Command is argv, never a shell string
- Tests green before deploy
- Smoke / bench only on idle GPUs
