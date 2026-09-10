# Policy

All knobs live in [`config/platform.yaml`](../config/platform.yaml). Nothing about hours or GPU caps is hardcoded.

Typical fields:

| Field | Meaning |
|-------|---------|
| `windows` | Named `{name,start,end,days,enabled}` rows (empty = always eligible). Overnight windows belong to the start weekday. |
| `max_troiani_gpus` | Cap on simultaneous Troiani GPUs |
| `max_jobs` | Cap on simultaneous Troiani jobs |
| `max_gpus_per_job` | Per-job width |
| `max_runtime_s` | Wall-clock limit before cooperative preempt |
| `preempt_grace_s` | Time allowed to finish a checkpoint |
| `cooldown_s` | Delay before reusing a just-released GPU |
| `memory_busy_gb` | Used-memory threshold that is not treated as idle |

Time windows never override a researcher. A GPU with a non-Troiani compute process is unavailable regardless of the clock.

`release-all` is the emergency stop: every opportunistic job is asked to checkpoint and leave.
