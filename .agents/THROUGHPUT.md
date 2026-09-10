# Token-ingestion throughput (same-node, 1 GPU)

Not M16. Not NCCL. 40GB ≠ 80GB. Seq default **2048**.
Probe: `python -m troiani_platform.training.throughput --params 1B --batch 4 --seq 2048 --steps 30`
(On a torch-only env, run the file path so `training/__init__.py` is not imported.)
Estimator + FIT gate: `src/troiani_platform/training/throughput.py`.
Job templates: `config/jobs/bench-throughput.yaml` (40GB) and `bench-throughput-80.yaml` (Uranus-only).
Command is an argv list. No `shell=True`. No large checkpoints (none written).

## Estimator (before launch)

Bytes ≈ params(bf16) + grads(bf16) + Adam m/v (fp32×2)
+ activations `batch × seq × hidden × layers × 2 × 8` (residual + MLP kept for backward; **not** full seq² attention)
+ CE logits `batch × seq × vocab × 4` + 1.5 GiB CUDA context.
FIT if total ≤ GPU GiB − 2 GiB reserve.

**Honesty:** the estimator is optimistic at large batch. Live 350M×16 used **39.7 GiB** on 80GB (est. 21). Same config **OOM on Atlas 40GB**. Trust live OOM over a FIT stamp.

Catalog (tied embed, vocab 32k):

| label | counted params | layers | hidden | heads |
|---|---:|---:|---:|---:|
| 350M | 334,858,240 | 24 | 1024 | 16 |
| 500M | 512,944,640 | 24 | 1280 | 16 |
| 777M | 785,415,168 | 26 | 1536 | 16 |
| 1B | 971,657,216 | 18 | 2048 | 16 |
| 1.5B (nearby) | 1,497,687,040 | 18 | 2560 | 20 |
| 2B | 1,969,607,680 | 24 | 2560 | 20 |

### Size × batch × {40, 80} GiB  (seq=2048)

| model | params | batch | est. GiB | 40GB | 80GB |
|---|---:|---:|---:|---|---|
| 350M | 334,858,240 | 1 | 6.24 | FIT | FIT |
| 350M | 334,858,240 | 2 | 7.23 | FIT | FIT |
| 350M | 334,858,240 | 4 | 9.22 | FIT | FIT |
| 350M | 334,858,240 | 8 | 13.20 | FIT | FIT |
| 350M | 334,858,240 | 16 | 21.15 | FIT on paper / **live OOM** | FIT |
| 500M | 512,944,640 | 1 | 8.41 | FIT | FIT |
| 500M | 512,944,640 | 2 | 9.60 | FIT | FIT |
| 500M | 512,944,640 | 4 | 11.96 | FIT | FIT |
| 500M | 512,944,640 | 8 | 16.69 | FIT | FIT |
| 500M | 512,944,640 | 16 | 26.14 | FIT on paper / untested live | FIT |
| 777M | 785,415,168 | 1 | 11.74 | FIT | FIT |
| 777M | 785,415,168 | 2 | 13.20 | FIT | FIT |
| 777M | 785,415,168 | 4 | 16.13 | FIT | FIT |
| 777M | 785,415,168 | 8 | 21.98 | FIT | FIT |
| 777M | 785,415,168 | 16 | 33.68 | FIT on paper / untested live | FIT |
| 1B | 971,657,216 | 1 | 13.73 | FIT | FIT |
| 1B | 971,657,216 | 2 | 15.10 | FIT | FIT |
| 1B | 971,657,216 | 4 | 17.84 | FIT | FIT |
| 1B | 971,657,216 | 8 | 23.31 | FIT | FIT |
| 1B | 971,657,216 | 16 | 34.27 | FIT on paper / untested live | FIT |
| 1.5B | 1,497,687,040 | 1 | 19.89 | FIT | FIT |
| 1.5B | 1,497,687,040 | 2 | 21.54 | FIT | FIT |
| 1.5B | 1,497,687,040 | 4 | 24.84 | FIT | FIT |
| 1.5B | 1,497,687,040 | 8 | 31.44 | FIT | FIT |
| 1.5B | 1,497,687,040 | 16 | 44.64 | **NO-FIT** | FIT |
| 2B | 1,969,607,680 | 1 | 25.63 | FIT | FIT |
| 2B | 1,969,607,680 | 2 | 27.75 | FIT | FIT |
| 2B | 1,969,607,680 | 4 | 31.99 | FIT | FIT |
| 2B | 1,969,607,680 | 8 | 40.47 | **NO-FIT** | FIT |
| 2B | 1,969,607,680 | 16 | 57.42 | **NO-FIT** | FIT |

Do not launch: **2B batch 8/16 on 40GB**, **1.5B batch 16 on 40GB**, and **350M batch 16 on 40GB** (live OOM).

## What actually ran (2026-09-10 19:42–19:45 Europe/Madrid)

Idle check before launch:

- Atlas gpu0: **gkoutr** 642620 ~26 GiB — not touched
- Atlas gpu1: idle → used for all Atlas rows
- Uranus gpu0: idle → used, then left (csp RepoEvolve 473920 appeared at ~532 MiB; stopped)
- Uranus gpu1: left free
- Uranus gpu2–3: **csp vLLM** 2524772/2524773 — not touched

Platform `.venv` has **no torch**. Timed steps used wtroi’s CUDA torch only (`miniconda3/envs/troiani` on Atlas, `.venv-dist` on Uranus). No researcher env. No worker restart. No `deploy.sh --delete`. Only `throughput.py` was rsynced.

Each timed row: 4 warmup + 24 steps, bf16, Adam, causal SDPA + CE, seq=2048.
`vram used` = `torch.cuda.max_memory_allocated` (GiB). `sm%` from nvidia-smi mid-run.

| model | params | batch | seq | node | gpu_ref | sm% | vram used | tokens/s | fit? |
|---|---:|---:|---:|---|---|---:|---:|---:|---|
| 350M | 334,858,240 | 1 | 2048 | atlas | atlas/gpu1 | 100 | 4.25 | 47526 | FIT |
| 350M | 334,858,240 | 4 | 2048 | atlas | atlas/gpu1 | 100 | 11.35 | 59945 | FIT |
| 350M | 334,858,240 | 8 | 2048 | atlas | atlas/gpu1 | 100 | 20.82 | 65354 | FIT |
| 350M | 334,858,240 | 16 | 2048 | atlas | atlas/gpu1 | — | — | — | **OOM** (est. said FIT) |
| 500M | 512,944,640 | 1 | 2048 | atlas | atlas/gpu1 | 100 | 5.63 | 35697 | FIT |
| 777M | 785,415,168 | 1 | 2048 | atlas | atlas/gpu1 | 100 | 7.73 | 26794 | FIT |
| 1B | 971,657,216 | 1 | 2048 | atlas | atlas/gpu1 | 100 | 8.57 | 22739 | FIT |
| 1B | 971,657,216 | 4 | 2048 | atlas | atlas/gpu1 | 100 | 17.94 | 29027 | FIT |
| 2B | 1,969,607,680 | 1 | 2048 | atlas | atlas/gpu1 | 100 | 15.67 | 12575 | FIT |
| 350M | 334,858,240 | 1 | 2048 | uranus | uranus/gpu0 | 100 | 4.25 | 50458 | FIT |
| 350M | 334,858,240 | 16 | 2048 | uranus | uranus/gpu0 | 100 | 39.74 | 72101 | FIT 80GB |
| 500M | 512,944,640 | 1 | 2048 | uranus | uranus/gpu0 | 100 | 5.63 | 36693 | FIT |
| 1B | 971,657,216 | 1 | 2048 | uranus | uranus/gpu0 | 100 | 8.57 | 23889 | FIT |
| 1B | 971,657,216 | 8 | 2048 | uranus | uranus/gpu0 | 100 | 30.43 | 32103 | FIT |
| 2B | 1,969,607,680 | 1 | 2048 | uranus | uranus/gpu0 | 100 | 15.67 | 13216 | FIT |
| 2B | 1,969,607,680 | 4 | 2048 | uranus | uranus/gpu0 | 100 | 29.55 | 16246 | FIT |

40GB vs 80GB at the same shape is close (1B×1: 22.7k vs 23.9k tok/s). Batch helps more than GPU size at these widths. 350M×16 only fits the 80GB card.

## Approximate time-to-budget (1 GPU, probe rate)

`always-on hours = tokens / tokens_per_sec / 3600`  
`policy hours = always-on / duty_cycle`

**Live cluster duty cycle = 100%.** `config/platform.yaml` windows (weeknights, lunch, weekends, …) are all `enabled: false`. All-disabled = opportunistic any hour. Windows were **not** turned on.

Hypothetical duty cycles from the same window code (Europe/Madrid, minute sample over 7 days). Overnight windows belong to the start weekday:

| scenario | hours / 168 | duty | used? |
|---|---:|---:|---|
| live / all-disabled | 168 / 168 | **100%** | yes (current policy) |
| weeknights only (Mon–Fri 20:00–08:00) | 60.08 / 168 | **35.8%** | hypothetical (user’s 12h×5/168 ≈ 35.7%) |
| weeknights ∪ weekends (Sat–Sun 00:00–23:59) | 100.07 / 168 | **59.6%** | hypothetical (Sat 00:00–08:00 counted once) |

These ETAs assume the short probe tok/s holds forever on one idle GPU, no preemption, no data-load stall, no checkpoint pause. Real wall time is longer. Not M16.

### 1B (the size they care about)

| gpu_ref | batch | tok/s | budget | always-on (100%) | weeknights (35.8%) | nights+weekend (59.6%) |
|---|---:|---:|---|---|---|---|
| atlas/gpu1 | 1 | 22,739 | 20B | 10 d | 28 d | 17 d |
| atlas/gpu1 | 1 | 22,739 | 100B | **51 d** | **142 d** | 85 d |
| atlas/gpu1 | 1 | 22,739 | 300B | 153 d | 1.17 y | 256 d |
| atlas/gpu1 | 1 | 22,739 | 1T | 1.39 y | 3.90 y | 2.34 y |
| atlas/gpu1 | 4 | 29,027 | 20B | 8.0 d | 22 d | 13 d |
| atlas/gpu1 | 4 | 29,027 | 100B | **40 d** | **111 d** | 67 d |
| atlas/gpu1 | 4 | 29,027 | 300B | 120 d | 334 d | 201 d |
| atlas/gpu1 | 4 | 29,027 | 1T | 1.09 y | 3.05 y | 1.83 y |
| uranus/gpu0 | 1 | 23,889 | 20B | 9.7 d | 27 d | 16 d |
| uranus/gpu0 | 1 | 23,889 | 100B | **48 d** | **135 d** | 81 d |
| uranus/gpu0 | 1 | 23,889 | 300B | 145 d | 1.11 y | 244 d |
| uranus/gpu0 | 1 | 23,889 | 1T | 1.33 y | 3.71 y | 2.23 y |
| uranus/gpu0 | 8 | 32,103 | 20B | 7.2 d | 20 d | 12 d |
| uranus/gpu0 | 8 | 32,103 | 100B | **36 d** | **101 d** | 61 d |
| uranus/gpu0 | 8 | 32,103 | 300B | 108 d | 302 d | 182 d |
| uranus/gpu0 | 8 | 32,103 | 1T | 361 d | 2.76 y | 1.66 y |

Chinchilla 20B on a 1B is ~1–1.5 weeks always-on at this probe. 100B is ~5–7 weeks always-on, ~3.5 months weeknights-only. 300B–1T (the regime they actually want) is months to years on **one** opportunistic GPU.

### Every benched size (100B / 300B / 1T)

| model | gpu_ref | batch | tok/s | 100B always | 100B weeknights | 300B always | 300B weeknights | 1T always | 1T weeknights |
|---|---|---:|---:|---|---|---|---|---|---|
| 350M | atlas/gpu1 | 1 | 47,526 | 24 d | 68 d | 73 d | 204 d | 244 d | 1.86 y |
| 350M | atlas/gpu1 | 8 | 65,354 | 18 d | 50 d | 53 d | 149 d | 177 d | 1.36 y |
| 350M | uranus/gpu0 | 1 | 50,458 | 23 d | 64 d | 69 d | 192 d | 229 d | 1.76 y |
| 350M | uranus/gpu0 | 16 | 72,101 | 16 d | 45 d | 48 d | 135 d | 161 d | 1.23 y |
| 500M | atlas/gpu1 | 1 | 35,697 | 32 d | 91 d | 97 d | 272 d | 324 d | 2.48 y |
| 500M | uranus/gpu0 | 1 | 36,693 | 32 d | 88 d | 95 d | 265 d | 315 d | 2.41 y |
| 777M | atlas/gpu1 | 1 | 26,794 | 43 d | 121 d | 130 d | 362 d | 1.18 y | 3.31 y |
| 1B | atlas/gpu1 | 1 | 22,739 | 51 d | 142 d | 153 d | 1.17 y | 1.39 y | 3.90 y |
| 1B | atlas/gpu1 | 4 | 29,027 | 40 d | 111 d | 120 d | 334 d | 1.09 y | 3.05 y |
| 1B | uranus/gpu0 | 1 | 23,889 | 48 d | 135 d | 145 d | 1.11 y | 1.33 y | 3.71 y |
| 1B | uranus/gpu0 | 8 | 32,103 | 36 d | 101 d | 108 d | 302 d | 361 d | 2.76 y |
| 2B | atlas/gpu1 | 1 | 12,575 | 92 d | 257 d | 276 d | 2.11 y | 2.52 y | 7.05 y |
| 2B | uranus/gpu0 | 1 | 13,216 | 88 d | 245 d | 263 d | 2.01 y | 2.40 y | 6.70 y |
| 2B | uranus/gpu0 | 4 | 16,246 | 71 d | 199 d | 214 d | 1.64 y | 1.95 y | 5.45 y |

After leave: Atlas gpu1 1 MiB; Uranus gpu0 is csp 473920 (not ours); gkoutr 642620 and vLLM 2524772/2524773 still there.

## Time to token budgets (always-on vs weeknights policy)

Live cluster windows are **all disabled** → duty cycle 100% (same as always-on) until you turn a rule on.

Hypothetical **weeknights** (Mon–Fri 20:00–08:00 Europe/Madrid): 60 h / 168 h = **35.7%** duty. Policy calendar time ≈ always-on / 0.357.

These use **measured step-loop tok/s** (bf16 + Adam + CE, seq 2048). No dataloader, checkpoint I/O, eval, or preemption. Real Troiani training will be slower.

| setup | tok/s | 20B always / weeknights | 100B always / weeknights | 300B always / weeknights | 1T always / weeknights |
|---|---:|---|---|---|---|
| 350M×8 atlas/gpu1 | 65354 | 3.5 d / 9.9 d | **17.7 d / 49.6 d** | 53.1 d / 149 d | 177 d / 496 d |
| 1B×1 atlas/gpu1 | 22739 | 10.2 d / 28.5 d | **50.9 d / 142 d** | 153 d / 427 d | 509 d / 3.9 y |
| 1B×4 atlas/gpu1 | 29027 | 8.0 d / 22.3 d | **39.9 d / 112 d** | 120 d / 335 d | 399 d / 3.1 y |
| 1B×1 uranus/gpu0 | 23889 | 9.7 d / 27.1 d | **48.4 d / 136 d** | 145 d / 407 d | 484 d / 3.7 y |
| 1B×8 uranus/gpu0 | 32103 | 7.2 d / 20.2 d | **36.1 d / 101 d** | 108 d / 303 d | 361 d / 2.8 y |
| 2B×1 atlas/gpu1 | 12575 | 18.4 d / 51.6 d | **92.1 d / 258 d** | 276 d / 773 d | 2.5 y / 7.1 y |
| 2B×4 uranus/gpu0 | 16246 | 14.3 d / 39.9 d | **71.3 d / 199 d** | 214 d / 598 d | 1.9 y / 5.5 y |

If the objective is a **good 1B** (≈300B–1T quality tokens, not Chinchilla 20B): one idle 40GB card at 1B×4 is ~4 months always-on to 300B, or ~11 months if you only train weeknights. Two idle cards (atlas/gpu1 + uranus/gpu0) in parallel would roughly halve that if they do not share one job (not NCCL).

## Pending

- Uranus 777M (csp took gpu0 after 500M; not launched)
- 1B×16 and 2B×8/16 on 80GB (not launched; 2B×8/16 **NO-FIT on 40GB**)
- 500M/777M batch sweep (only batch 1 timed)
- Do not install torch into the shared platform venv unless an operator asks
- M16 real 1B pretrain: still later
