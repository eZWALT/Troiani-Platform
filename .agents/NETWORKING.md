# GPU interconnect (read-only facts, 2026-09-10)

Question: *why PCIe and not NVLink/NVSwitch — missing hardware, or we just do not use it?*

**Short answer:** Atlas has **no active NVLink** (two PCIe A100 40GB, topo `NODE`). Uranus **does** have NVLink, but only as **two pairs** (`NV12` GPU0–1 and GPU2–3), not an all-to-all NVSwitch fabric. There is **no NVLink or InfiniBand between Atlas and Uranus** (two chassis, 10GbE). The platform control plane is HTTP on that LAN; that is orthogonal to on-node NVLink. v1 jobs are same-node because tensors never cross the chassis boundary. Troiani does not use the NVLink that Uranus already has (dummy trainer, 1-GPU benches, no NCCL).

Do **not** “enable NVSwitch”. These are PCIe cards, not DGX/HGX SXM boards. `lspci` has no NVSwitch device on either node.

Collected over SSH (`nvidia-smi`, `lspci`, `/proc/driver/nvidia`, `ip`/`ethtool`, no IB). Researcher PIDs were not touched.

## Atlas `192.168.1.17` — 2× A100-PCIE-40GB

| | GPU0 | GPU1 |
|---|---|---|
| Product (`nvidia-smi -L`) | `NVIDIA A100-PCIE-40GB` | `NVIDIA A100-PCIE-40GB` |
| Form factor | PCIe (`Bus Type: PCIe`), `MultiGPU Board: No` | same |
| lspci | `b1:00.0 GA100 [A100 PCIe 40GB]` | `ca:00.0 GA100 [A100 PCIe 40GB]` |
| UUID | `GPU-5d8aba68-7076-7d7c-e1d6-16ef58cb7b20` | `GPU-4497e145-7ee3-dc4e-5227-a824fc72d9a6` |
| VBIOS | `92.00.25.00.08` | `92.00.90.00.08` |
| Board P/N | `900-21001-0000-000` | `900-21001-0000-000` |
| GPU P/N | `20F1-883-A1` | (same product family) |
| PCI | `00000000:B1:00.0` | `00000000:CA:00.0` |
| PCIe | Gen **4** / **16x** (max = current; host max 4) | Gen **4** / **16x** |
| NUMA | node **1** (CPUs 28–55, 84–111) | node **1** |
| Driver / CUDA | 565.57.01 / 12.7 | same |

CPU: 2× Intel Xeon Gold 6348, 112 threads, 2 NUMA nodes. Both GPUs sit on NUMA 1.

### Topology matrix (`nvidia-smi topo -m`)

```
	GPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	NODE	28-55,84-111	1		N/A
GPU1	NODE	 X 	28-55,84-111	1		N/A
```

Legend (NVIDIA): `NODE` = PCIe plus the interconnect between PCIe host bridges **within a NUMA node**. Not `NV#`.

### NVLink status

```
nvidia-smi nvlink -s
GPU 0: NVIDIA A100-PCIE-40GB
NVML: Unable to retrieve NVLink information as all links are inActive
GPU 1: NVIDIA A100-PCIE-40GB
NVML: Unable to retrieve NVLink information as all links are inActive
```

`nvidia-smi nvlink -c` / `-p` print the two GPU names and **no link rows**. No peers.

So GPU0↔GPU1 on Atlas is **PCIe only**. NVML’s “all links are inActive” means there is no up NVLink (no bridge / no connected links). This is not an SXM/NVSwitch box. Do not plan on flipping a software switch to get NVSwitch here.

## Uranus `192.168.1.18` — 4× A100 80GB PCIe

| | GPU0 | GPU1 | GPU2 | GPU3 |
|---|---|---|---|---|
| Product | `NVIDIA A100 80GB PCIe` | same | same | same |
| Form factor | PCIe, `MultiGPU Board: No` | same | same | same |
| lspci | `27:00.0 A100 PCIe 80GB` | `38:00.0` | `a8:00.0` | `b8:00.0` |
| UUID | `GPU-e24787d5-…` | `GPU-5bd670ab-…` | `GPU-c5934fc0-…` | `GPU-b6f04d55-…` |
| VBIOS | `92.00.A0.00.05` | same | same | same |
| Board P/N | `900-21001-0020-100` | same | same | same |
| GPU P/N | `20B5-893-A1` | same | same | same |
| PCI | `00000000:27:00.0` | `00000000:38:00.0` | `00000000:A8:00.0` | `00000000:B8:00.0` |
| PCIe | Gen **4** / **16x** (device max 4; **host max 5**) | same | same | same |
| NUMA | **0** | **0** | **1** | **1** |
| Driver / CUDA | 610.43.02 / 13.3 | same | same | same |

CPU: 2× Intel Xeon Platinum 8470, 208 threads, 2 NUMA nodes.

### Topology matrix (`nvidia-smi topo -m`)

```
	GPU0	GPU1	GPU2	GPU3	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	NV12	SYS	SYS	0-51,104-155	0		N/A
GPU1	NV12	 X 	SYS	SYS	0-51,104-155	0		N/A
GPU2	SYS	SYS	 X 	NV12	52-103,156-207	1		N/A
GPU3	SYS	SYS	NV12	 X 	52-103,156-207	1		N/A
```

`NV12` = twelve bonded NVLinks. `SYS` = PCIe plus the **SMP/UPI path between NUMA nodes**.

### NVLink status (`nvidia-smi nvlink -s`)

Every GPU reports **12 links at 25 GB/s**. `nvidia-smi nvlink -p` peers:

- GPU0 links 0–11 → `00000000:38:00.0` (GPU1)
- GPU1 links 0–11 → `00000000:27:00.0` (GPU0)
- GPU2 links 0–11 → `00000000:B8:00.0` (GPU3)
- GPU3 links 0–11 → `00000000:A8:00.0` (GPU2)

Capabilities (`nvlink -c`): P2P, sysmem access, P2P atomics, sysmem atomics, link supported = true on those links.

This is the **2-GPU NVLink bridge** pattern on PCIe A100s: two isolated pairs, not a 4-GPU NVSwitch mesh. GPU0 cannot reach GPU2 over NVLink.

`lspci | grep -i nvswitch` — no device. No `/dev/infiniband`, no `ibstat`.

## Bandwidth (order of magnitude)

| Path | What it is here | ≈ unidirectional |
|---|---|---|
| PCIe Gen3 x16 | not what these cards are running | ~16 GB/s |
| PCIe Gen4 x16 | Atlas GPU↔CPU and Atlas GPU0↔GPU1 (`NODE`); Uranus GPU↔CPU; Uranus cross-pair (`SYS` also crosses UPI) | ~32 GB/s |
| NVLink 3.0 ×12 (`NV12`) | Uranus GPU0↔1 and GPU2↔3 only | 12 × 25 GB/s = **300 GB/s** (600 GB/s bidirectional) |
| NVSwitch all-to-all | **absent** (DGX/HGX SXM fabric) | n/a |
| Lab NIC `eno1` | Atlas and Uranus: `ethtool` **10000Mb/s** full duplex, twisted pair | 1.25 GB/s |
| InfiniBand | **absent** on both | n/a |

PCIe Gen4 x16 is the intra-node path on Atlas and the **cross-NUMA** path on Uranus. NVLink is ~10× that, but only inside each Uranus pair.

## Control plane vs GPU fabric

Workers POST heartbeats, incremental logs (32 KiB chunks), and checkpoint **manifests** to `http://192.168.1.17:8787`. That is HTTP over the **10GbE LAN**. Training tensors do **not** ride that path in v1.

That HTTP hop is why Atlas↔Uranus feels like “the network”. It has nothing to do with whether NVLink is up on a node. You would still use HTTP for the control plane on a DGX.

## Why v1 jobs are same-node

Placement picks GPUs on **one** `node` (`placement.py`). There is no NCCL launch across hosts.

Even if both boxes had NVSwitch internally, **there is no NVLink between Atlas and Uranus**. Two chassis, 10GbE, no IB. A 2-node Troiani “smoke” is two independent jobs, not one allreduce.

## Do we have NVLink and not use it?

| Node | Hardware | Does Troiani use it? |
|---|---|---|
| Atlas GPU0–1 | **No active NVLink.** PCIe `NODE` only. | N/A — the path is not there. |
| Uranus GPU0–1 | **Yes, NV12**, 12×25 GB/s. | **No.** Dummy smokes are CPU+sleep. Throughput benches were **1 GPU**. No NCCL. |
| Uranus GPU2–3 | **Yes, NV12** (often `csp` vLLM — leave them). | Same: platform does not NCCL. |
| Uranus 0/1 ↔ 2/3 | **No NVLink** (`SYS`). | Same-node 4-GPU NCCL would still traverse UPI/PCIe between pairs. |
| Atlas ↔ Uranus | **No GPU interconnect.** 10GbE only. | Must not NCCL across nodes. |

If a **same-node** multi-GPU Troiani job on Uranus GPU0+1 actually runs NCCL, the runtime will use NVLink without any “enable NVSwitch” step. That is a P2 when such a job exists — not for 2-node, and not for dummy smokes.

## Access

We have access to `nvidia-smi` on both machines as `wtroi`. We are not missing a secret NVSwitch. Atlas simply has no up NVLink. Uranus has pair NVLink and we are not exercising it yet.
