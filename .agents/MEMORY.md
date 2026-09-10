# Troiani-Platform agent memory

Updated: 2026-09-10 19:20 Europe/Madrid

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
Subtitle: `Atlas · Uranus · opportunistic`.
Policy: emergency STOP ALL / Resume, aggressiveness buttons, rule chips, current vs proposed, On/Off partial stop.
Cluster GPU cards show SM and VRAM separately.

## Invariants

- No researcher PID kill
- Do not enable overnight/weekend windows live unless the user turns them on
- Command is argv, never a shell string
- Tests green before deploy
- Smoke only on idle GPUs
