# Pretraining tokens vs parameters (user guidance)

Persist this as meaning, not a slogan. Re-read before anyone proposes a “Chinchilla-optimal” 1B run.

## Compute-optimal is not “good enough”

For a **~1B dense** model, Chinchilla’s **~20 tokens per parameter** is **compute-optimal** (the cheap way to spend a fixed FLOP budget). It is **not** the bar for a model that is actually good.

## Token table they care about (1B-class)

| tokens | multiple | reading |
|---|---|---|
| 10B | 10× | undertrained |
| 20B | 20× | Chinchilla-optimal for **training compute** — not “done” |
| 50B | 50× | solid small LM |
| 100B | 100× | quite good if data and arch are good |
| 300B | 300× | very strong for 1B |
| 1T | 1000× | seriously overtrained but potentially excellent |
| 3T | 3000× | TinyLlama-style extreme |
| 5–10T | 5–10k× | research / diminishing returns |

**Interesting regime: 100B–1T.**

If training a **1B today for cheap inference**: target **≈300B–1T high-quality tokens**, not blindly 20B.

## Data quality beats the token counter

Once you are in the **hundreds of billions**, data quality matters more than the raw counter.

Prefer **1B × 500B excellent tokens** over **1B × 3T garbage**.

## What this file is not

This is **not** a green light for M16 (multi-day real Troiani 1B pretrain). M16 stays later, idle GPUs only.

Throughput / VRAM fit numbers live in `THROUGHPUT.md`. Same-node 1 GPU. 40GB ≠ 80GB. Not NCCL.
