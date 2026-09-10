# Storage and what actually bounds the smokes (2026-09-10)

Infra is fixed (Atlas 2×40GB, Uranus 4×80GB, 10GbE between boxes). Gains here are **config / software**, not “buy NVSwitch”. See [NETWORKING.md](NETWORKING.md) for the GPU fabric.

Three workloads, do not mix them:

1. **`dummy_train` smokes** — CPU loop + `time.sleep` + tiny pickle checkpoints + HTTP metrics every step. **Not GPU-compute-bound.**
2. **`throughput.py` benches** (350M–2B, seq 2048, bf16+Adam) — SM was 100% in the live rows. Synthetic step loop, **no checkpoints**, no dataloader. Same-node, 1 GPU, no NCCL.
3. **Control-plane path** — heartbeats, 32 KiB log chunks, checkpoint **manifests** POST to Atlas. Weight bytes stay worker-local.

## Where bytes live

| What | Where | Notes |
|---|---|---|
| Checkpoint files (`state.pkl`, `manifest.json`, `VALID`) | `~/Troiani-Platform/var/checkpoints` **on the worker node** | Atlas jobs → Atlas disk. Uranus jobs → Uranus disk. |
| Checkpoint metadata | Atlas SQLite `var/platform.db` | Workers POST `/v1/internal/checkpoint`. Path string points at the **remote** dir. |
| Job / run / GPU state | Atlas `var/platform.db` (~1.0 MiB today) | Control plane of record. |
| MLflow | Atlas `var/mlflow.db` (~1.1 MiB), `var/mlruns` | Loopback `:5000`. |
| TensorBoard | Atlas `var/tensorboard` | Loopback `:6006`. |
| Worker / control logs | each node `var/logs` | Incremental tails (≤32 KiB / heartbeat) copied to Atlas KV `log_tail:*`. |
| Training tensors | GPU HBM on that node | Never cross the LAN in v1. |

**Atlas-as-store is not implemented.** Do not silently mount NFS/`/data` under `var/checkpoints`. If someone wants a shared store later, that is an explicit operator choice (and a different design).

### Disk facts (read-only)

**Atlas** `findmnt -T ~/Troiani-Platform/var/checkpoints`:

```
/home  /dev/nvme1n1p1  ext4  rw,relatime,stripe=32
```

`df -h /home`: 7.0T, 2.2T used, 4.5T avail (33%). `lsblk`: `nvme1n1` Samsung MZQL27T6HBLA-00A07, **ROTA=0**, NVMe. Not NFS. (HDD RAID `/data` and SATA SSD `/` exist; Troiani `var/` is **not** on them.)

`du -sh` on Atlas (do not delete):

```
180K   var/checkpoints
836K   var/logs
4.0K   var/mlruns
1.1M   var/mlflow.db
1008K  var/platform.db
```

**Uranus** `findmnt -T ~/Troiani-Platform/var/checkpoints`:

```
/home  home  zfs  rw,xattr,noacl
```

ZFS pool `home`: **raidz1 of four NVMe** (`nvme2n1`–`nvme5n1`, Samsung MZQL27T6HBLA), 27.9T pool, `/home` 6.66T used / 13.5T avail, compression off, recordsize 128K. Not NFS. The spinning TOSHIBA 18T raidz1 is pool `data` → `/data` (unused by Troiani `var/`).

`du -sh` on Uranus: `var/checkpoints` 601K, `var/logs` 71K.

### Checkpoint code path

`CheckpointManager.save`: write `*.tmp`, `fsync` files + dir, validate (sha256 + reload), `os.rename` onto the final id, `VALID` marker, then GC. `AsyncCheckpointWriter`: **one in-flight** regular save in a child process; preemption/shutdown/failure stay **synchronous**.

Dummy `state.pkl` on both nodes: **40–56 bytes** (pickle of `{step, rng, loss}`).

A **1B Adam** state is a different species. Counted params `971,657,216` ([THROUGHPUT.md](THROUGHPUT.md)):

- bf16 weights: ×2 ≈ 1.81 GiB
- Adam m+v fp32: ×8 ≈ 7.24 GiB
- **≈ 9.05 GiB** if that is what you serialize (plus `torch.save` overhead). Grads usually not stored.

Dummy vs 1B: **~50 B vs ~10 GiB** (~2×10⁸×). Smoke checkpoint I/O does not predict pretrain I/O.

## Dummy smoke path (why it is not a GPU test)

`dummy_train` (`config/jobs/smoke.yaml`: `--steps 60 --sleep 0.1`, `checkpoint_every_steps: 20`):

- No CUDA. Loss is `1/sqrt(step)`.
- `time.sleep(args.sleep)` every step → **sleep-bound** wall clock (~6 s + HTTP).
- `_post(..., "/v1/internal/metrics")` **every step** (`urllib`, 5 s timeout). From Uranus that is a LAN RTT+SQLite write per step.
- Regular ckpt every `CHECKPOINT_EVERY_STEPS` (env, default 25) via async writer; still spawn + fsync + manifest POST.
- Printed `tok/s` is `tokens_per_step / sleep`, **fake**.

So a 2-node smoke is two independent sleep loops. Gradients do not hit 10GbE. What *does* hit the LAN: metrics POSTs, heartbeats (`policy.worker_heartbeat_s` = 10), log chunks, tiny manifests. That can make a chatty smoke **control/HTTP-bound** if you drop `--sleep` toward 0.

## Throughput benches (live 2026-09-10)

Rows in THROUGHPUT.md: SM **100%** mid-run, `torch.cuda.max_memory_allocated` for VRAM, 4 warmup + 24 steps, no ckpt, no host dataloader.

| Observation | Bound |
|---|---|
| SM 100% even at batch 1 | SMs are busy. **Not** “idle GPU”. SM% still does not prove math-bound vs HBM-bound. |
| tok/s **rises** with batch (350M Atlas 47.5k→59.9k→65.4k at b1/4/8; 1B Atlas 22.7k→29.0k at b1/4; 1B Uranus 23.9k→32.1k at b1/8) | Small batch is **under-occupying** (latency / small GEMMs). Larger batch buys tokens/s. |
| Diminishing returns (350M 4→8 only +9%) | Moving toward a **compute or HBM plateau**, not more free SM. |
| VRAM climbs with batch (350M Atlas 4.25→20.82 GiB; 350M×16 **OOM** on 40GB, **39.74 GiB** on 80GB) | Large batch is **activation / capacity** bound. Estimator was optimistic. |
| 40GB vs 80GB tok/s almost equal at the same shape (1B×1: 22.7k vs 23.9k) | Extra HBM does not speed the step when the working set fits. Same A100-class SMs. |
| No ckpt, synthetic inputs | **Not** disk-bound, **not** PCIe host↔device-bound, **not** LAN-bound. |
| 1 GPU, no NCCL | On-node NVLink unused (Uranus) / absent (Atlas). |

Honest label for the 1B bench: **compute + activation-memory**, occupancy-limited at batch 1, **capacity-OOM** if you copy 350M×16 onto 40GB. Not a PCIe-vs-NVLink story.

## Bound table

| Workload | Compute (SM) | HBM / VRAM | PCIe GPU | Disk | LAN HTTP |
|---|---|---|---|---|---|
| Dummy 1-GPU smoke | no (CPU+sleep) | no | no | no (40–56 B pickle) | **yes if chatty** (metrics/step + heartbeat) |
| Dummy 2-node smoke | no | no | no | no | **control path only**; **not** gradient interconnect |
| 1B bench batch 1 | **yes** (SM 100%, still occupancy-limited) | working set fits (~8.6 GiB) | no (no H2D stream) | no (no ckpt) | no |
| 1B bench larger batch | yes, tok/s up | VRAM up (17.9 GiB @ b4 / 30.4 GiB @ b8) | no | no | no |
| 350M×16 on 40GB | n/a | **OOM (capacity)** | n/a | n/a | n/a |
| Future 1B pretrain, ckpt every 20 steps | step loop | activations | optional dataloader | **yes (~10 GiB × fsync+validate)** | manifests only |
| Future 1B pretrain, ckpt every 1k–2k steps | step loop | activations | dataloader if real data | NVMe can absorb | manifests only |

## Config knobs that actually help (this hardware)

Do **not** enable NCCL across Atlas and Uranus.

**Dummy / platform smokes**

- `CHECKPOINT_EVERY_STEPS` / `checkpoint_every_steps` — 20 is fine for proving save+POST. Do not reuse 20 for a real 1B.
- `--sleep` — this **is** the smoke duration. Lowering it makes HTTP to Atlas louder, not the GPU busier.
- Metrics POST **batching** (every 5–10 steps or ~1 s) — biggest software win if smokes look “slow” on Uranus. Today every step hits `/v1/internal/metrics`.

**Real-ish / 1B-class (software, same boxes)**

- **Batch 4 on Atlas 40GB** (measured 29.0k vs 22.7k tok/s). **Batch 8 on Uranus 80GB** (32.1k vs 23.9k). Do not launch 350M×16 on 40GB.
- Seq 2048 is the documented default; raising seq burns VRAM as activations.
- Grad accum if you want a larger *effective* batch without the 40GB OOM wall.
- Checkpoint interval in the **hundreds–thousands of steps**, not 20. Async writer already exists (one in-flight; emergency saves sync).
- Same-node only. Two idle cards = two jobs, not one NCCL job.

### Three concrete changes worth making

1. **Batch dummy metrics** — post `/v1/internal/metrics` every N steps (N=5–10) instead of every step. Smokes stay correct; Uranus stops chatting the control plane per sleep tick.
2. **Split smoke vs pretrain checkpoint cadence** — keep smoke at 20 steps; put a real 1B job at ≥500–2000 steps (or time-based, e.g. 10–30 min). Same YAML field, different value.
3. **Use the measured batch on a 1B** — Atlas `batch=4` on the idle 40GB card; Uranus `batch=8` only on an idle 80GB card. Leave `csp` / `gkoutr` GPUs alone. Still 1 GPU, still no cross-node NCCL.

## When a future 1B pretrain becomes disk-bound

NVMe on both homes is fast enough for **occasional** ~10 GiB atomic writes. It is not fast enough (and validate is not cheap) if you copy the smoke interval:

- 1B×4 Atlas ≈ 29k tok/s, 8192 tok/step → ~3.5 steps/s.
- Every **20** steps ≈ every 6 s, write ~10 GiB + sha256 + reload → the step loop waits on disk/CPU (async writer only hides **one** save; the next regular save `wait()`s).
- Every **2000** steps ≈ every ~10 min → a few hundred MiB/s average, comfortable on these NVMes (Atlas ext4 NVMe; Uranus raidz1 NVMe, write amp from parity, still fine).

Retention (`last_regular: 5`, `best_n: 3`, latest preempt/healthy) caps how many ~10 GiB trees you keep **per run**, on **that node**. Atlas disk GC cannot delete Uranus files.

LAN 10GbE would become the bound only if someone shipped full checkpoints to Atlas every save. That is Atlas-as-store. It is not built. Do not turn it on by accident via NFS.
