"""Same-node token-ingestion throughput probe + VRAM fit estimator.

This is not M16 (multi-day Troiani 1B pretrain). It measures tokens/s of a
decoder-shaped step (attention + MLP + Adam) at a documented size catalog.

Default sequence length: 2048.

Estimator (honest enough to mark FIT / NO-FIT before launch):
  params (bf16) + grads (bf16) + Adam m/v (fp32×2)
  activations ≈ batch * seq * hidden * layers * 2 bytes * 8
  (the ×8 covers residual + MLP intermediates kept for backward; not full seq² attention)
  plus vocab logits in fp32 for the CE stand-in, plus 1.5 GiB CUDA context.
  Live note: large-batch is optimistic (350M×16 estimated 21 GiB, used ~40 GiB).

Timed steps run only when CUDA is present and the estimator says FIT.
No NCCL. No cross-node. No large checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Iterable


def gpu_ref(node: str | None, index: int | None, uuid: str | None = None) -> str:
    host = (node or "node").strip().lower() or "node"
    if index is None and uuid:
        return f"{host}/{str(uuid)[:8]}"
    return f"{host}/gpu{int(index or 0)}"

DEFAULT_SEQ = 2048
DEFAULT_STEPS = 30
DEFAULT_WARMUP = 5
DEFAULT_BATCHES = (1, 2, 4, 8, 16)
DEFAULT_GPU_GIB = (40, 80)
DEFAULT_SIZES = ("350M", "500M", "777M", "1B", "2B")
NEARBY_SIZES = ("1.5B",)
RESERVE_GIB = 2.0
CUDA_OVERHEAD_GIB = 1.5
PARAM_BYTES = 2  # bf16 weights
GRAD_BYTES = 2  # bf16 grads
ADAM_BYTES = 8  # two fp32 states
# Residual stream + MLP intermediates saved for backward. Not seq² attention.
ACT_FACTOR = 8
GIB = 1024**3

_preempt = False
_shutdown = False


def _handle_preempt(_signum: int, _frame: object | None) -> None:
    global _preempt
    _preempt = True


def _handle_stop(_signum: int, _frame: object | None) -> None:
    global _shutdown
    _shutdown = True


@dataclass(frozen=True)
class ModelArch:
    name: str
    n_layers: int
    hidden: int
    n_heads: int
    vocab: int = 32000
    mlp_mult: int = 4

    @property
    def n_params(self) -> int:
        return count_decoder_params(self)

    @property
    def head_dim(self) -> int:
        return self.hidden // self.n_heads


@dataclass(frozen=True)
class VramEstimate:
    name: str
    n_params: int
    batch: int
    seq: int
    gpu_gb: float
    param_gib: float
    grad_gib: float
    adam_gib: float
    act_gib: float
    logits_gib: float
    overhead_gib: float
    total_gib: float
    usable_gib: float
    fits: bool

    @property
    def verdict(self) -> str:
        return "FIT" if self.fits else "NO-FIT"


# Decoder-style widths. Labels are classes (350M-class), counted params differ slightly.
# 350M ≈ GPT-2 medium / OPT-350M (24×1024). 1B ≈ 18×2048. 2B ≈ 24×2560.
# 1.5B is the nearby fallback when 2B is NO-FIT on 40GB.
CATALOG: dict[str, ModelArch] = {
    "350M": ModelArch("350M", n_layers=24, hidden=1024, n_heads=16),
    "500M": ModelArch("500M", n_layers=24, hidden=1280, n_heads=16),
    "777M": ModelArch("777M", n_layers=26, hidden=1536, n_heads=16),
    "1B": ModelArch("1B", n_layers=18, hidden=2048, n_heads=16),
    "1.5B": ModelArch("1.5B", n_layers=18, hidden=2560, n_heads=20),
    "2B": ModelArch("2B", n_layers=24, hidden=2560, n_heads=20),
}


def count_decoder_params(arch: ModelArch) -> int:
    """Tied-embedding decoder: embed + L × (attn + MLP + 2×LN) + final LN."""
    hidden = arch.hidden
    embed = arch.vocab * hidden
    attn = 4 * hidden * hidden  # qkv + out, no bias
    mlp = 2 * arch.mlp_mult * hidden * hidden
    lns = 4 * hidden  # two LayerNorms, weight+bias
    per_layer = attn + mlp + lns
    final_ln = 2 * hidden
    return embed + arch.n_layers * per_layer + final_ln


def parse_size(raw: str) -> ModelArch:
    key = str(raw).strip().upper().replace(" ", "")
    aliases = {
        "0.35B": "350M",
        "0.5B": "500M",
        "0.777B": "777M",
        "1.0B": "1B",
        "2.0B": "2B",
        "1500M": "1.5B",
    }
    key = aliases.get(key, key)
    if key not in CATALOG:
        known = ", ".join(CATALOG)
        raise ValueError(f"unknown size {raw!r}; known: {known}")
    return CATALOG[key]


def get_arch(name: str) -> ModelArch:
    return parse_size(name)


def _gib(num_bytes: float) -> float:
    return float(num_bytes) / GIB


def estimate_vram(
    arch: ModelArch,
    batch: int,
    seq: int = DEFAULT_SEQ,
    gpu_gb: float = 40.0,
    reserve_gib: float = RESERVE_GIB,
) -> VramEstimate:
    n_params = arch.n_params
    param = n_params * PARAM_BYTES
    grad = n_params * GRAD_BYTES
    adam = n_params * ADAM_BYTES
    # activations ≈ batch * seq * hidden * layers (bf16), × ACT_FACTOR for backward.
    act = int(batch) * int(seq) * arch.hidden * arch.n_layers * PARAM_BYTES * ACT_FACTOR
    logits = int(batch) * int(seq) * arch.vocab * 4
    overhead = int(CUDA_OVERHEAD_GIB * GIB)
    total = param + grad + adam + act + logits + overhead
    usable = max(0.0, float(gpu_gb) - float(reserve_gib))
    return VramEstimate(
        name=arch.name,
        n_params=n_params,
        batch=int(batch),
        seq=int(seq),
        gpu_gb=float(gpu_gb),
        param_gib=round(_gib(param), 3),
        grad_gib=round(_gib(grad), 3),
        adam_gib=round(_gib(adam), 3),
        act_gib=round(_gib(act), 3),
        logits_gib=round(_gib(logits), 3),
        overhead_gib=round(CUDA_OVERHEAD_GIB, 3),
        total_gib=round(_gib(total), 3),
        usable_gib=round(usable, 3),
        fits=_gib(total) <= usable,
    )


def estimate_fit(
    arch: ModelArch,
    batch: int,
    seq: int = DEFAULT_SEQ,
    gpu_gb: float = 40.0,
) -> VramEstimate:
    return estimate_vram(arch, batch=batch, seq=seq, gpu_gb=gpu_gb)


def estimate_matrix(
    sizes: Iterable[str] = DEFAULT_SIZES,
    batches: Iterable[int] = DEFAULT_BATCHES,
    gpu_sizes: Iterable[float] = DEFAULT_GPU_GIB,
    seq: int = DEFAULT_SEQ,
) -> list[VramEstimate]:
    rows: list[VramEstimate] = []
    for name in sizes:
        arch = parse_size(name)
        for batch in batches:
            for gpu_gb in gpu_sizes:
                rows.append(estimate_fit(arch, batch=int(batch), seq=seq, gpu_gb=float(gpu_gb)))
    return rows


def format_estimate_table(rows: Iterable[VramEstimate]) -> str:
    lines = [
        "| model | params | batch | seq | GPU GiB | est. GiB | usable | fit? |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.name} | {row.n_params:,} | {row.batch} | {row.seq} | "
            f"{row.gpu_gb:g} | {row.total_gib:.2f} | {row.usable_gib:.2f} | {row.verdict} |"
        )
    return "\n".join(lines)


def _hostname() -> str:
    return socket.gethostname().split(".")[0].strip().lower() or "node"


def _visible_index() -> int | None:
    raw = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not raw:
        return 0
    first = raw.split(",")[0].strip()
    if first.isdigit():
        return int(first)
    return None


def _read_smi(local_index: int | None) -> dict[str, float | int | None]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    if local_index is not None:
        cmd[1:1] = ["-i", str(local_index)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=8)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"index": local_index, "sm": None, "vram_used_gb": None, "vram_total_gb": None}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"index": local_index, "sm": None, "vram_used_gb": None, "vram_total_gb": None}
    line = proc.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    try:
        idx = int(float(parts[0]))
        sm = float(parts[1])
        used_mib = float(parts[2])
        total_mib = float(parts[3])
    except (IndexError, ValueError):
        return {"index": local_index, "sm": None, "vram_used_gb": None, "vram_total_gb": None}
    return {
        "index": idx,
        "sm": sm,
        "vram_used_gb": round(used_mib / 1024.0, 2),
        "vram_total_gb": round(total_mib / 1024.0, 2),
    }


def _post(endpoint: str, path: str, payload: dict) -> None:
    if not endpoint:
        return
    try:
        import urllib.request

        req = urllib.request.Request(
            endpoint.rstrip("/") + path,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('TROIANI_PLATFORM_TOKEN', '')}",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:
        pass


def _build_model(arch: ModelArch, device: object, dtype: object) -> object:
    from torch import nn
    from torch.nn import functional as F

    class DecoderBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_heads = arch.n_heads
            self.hidden = arch.hidden
            self.head_dim = arch.head_dim
            self.ln1 = nn.LayerNorm(arch.hidden, device=device, dtype=dtype)
            self.qkv = nn.Linear(
                arch.hidden, 3 * arch.hidden, bias=False, device=device, dtype=dtype
            )
            self.proj = nn.Linear(
                arch.hidden, arch.hidden, bias=False, device=device, dtype=dtype
            )
            self.ln2 = nn.LayerNorm(arch.hidden, device=device, dtype=dtype)
            self.fc1 = nn.Linear(
                arch.hidden, arch.mlp_mult * arch.hidden, bias=False, device=device, dtype=dtype
            )
            self.fc2 = nn.Linear(
                arch.mlp_mult * arch.hidden, arch.hidden, bias=False, device=device, dtype=dtype
            )

        def forward(self, x: object) -> object:
            h = self.ln1(x)
            qkv = self.qkv(h)
            batch, seq, _ = x.shape
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.hidden)
            x = x + self.proj(attn)
            h = self.ln2(x)
            return x + self.fc2(F.gelu(self.fc1(h)))

    class TinyDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(arch.vocab, arch.hidden, device=device, dtype=dtype)
            self.blocks = nn.ModuleList(DecoderBlock() for _ in range(arch.n_layers))
            self.ln_f = nn.LayerNorm(arch.hidden, device=device, dtype=dtype)

        def forward(self, tok: object) -> object:
            h = self.embed(tok)
            for block in self.blocks:
                h = block(h)
            h = self.ln_f(h)
            return F.linear(h, self.embed.weight)

    return TinyDecoder()


def _adam(model: object):
    import torch

    params = list(model.parameters())
    try:
        return torch.optim.Adam(params, lr=1e-4, fused=True)
    except (TypeError, RuntimeError):
        return torch.optim.Adam(params, lr=1e-4)


def run_probe(
    arch: ModelArch,
    batch: int,
    seq: int = DEFAULT_SEQ,
    steps: int = DEFAULT_STEPS,
    warmup: int = DEFAULT_WARMUP,
    gpu_gb: float | None = None,
) -> dict:
    """Timed decoder-style step. Refuses to allocate when the estimator says NO-FIT."""
    import torch
    from torch.nn import functional as F

    if not torch.cuda.is_available():
        return {
            "ok": False,
            "reason": "no CUDA",
            "tokens_per_sec": None,
            "fit": False,
            "timed": False,
        }

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    detected_gb = float(props.total_memory) / GIB
    target_gb = float(gpu_gb) if gpu_gb is not None else detected_gb
    estimate = estimate_fit(arch, batch=batch, seq=seq, gpu_gb=target_gb)
    node = _hostname()
    vis = _visible_index()
    ref = gpu_ref(node, vis if vis is not None else 0)
    base = {
        "ok": estimate.fits,
        "model": arch.name,
        "params": arch.n_params,
        "n_layers": arch.n_layers,
        "hidden": arch.hidden,
        "batch": batch,
        "seq": seq,
        "node": node,
        "gpu_ref": ref,
        "gpu_gb": round(detected_gb, 2),
        "estimate_gib": estimate.total_gib,
        "fit": estimate.fits,
        "verdict": estimate.verdict,
        "timed": False,
        "tokens_per_sec": None,
        "vram_used_gb": None,
        "sm": None,
        "steps": 0,
    }
    if not estimate.fits:
        base["reason"] = "estimator NO-FIT; refused to allocate"
        return base

    print(
        f"throughput start model={arch.name} batch={batch} seq={seq} gpu={ref}",
        flush=True,
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = _build_model(arch, device=device, dtype=dtype)
    opt = _adam(model)
    model.train()
    tokens = torch.randint(0, arch.vocab, (batch, seq), device=device)
    timed_steps = 0
    sm_sample: float | None = None
    t0 = 0.0
    try:
        for step in range(warmup + steps):
            if _preempt or _shutdown:
                break
            opt.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = F.cross_entropy(logits.float().reshape(-1, arch.vocab), tokens.reshape(-1))
            loss.backward()
            opt.step()
            if step + 1 == warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
            if step + 1 > warmup:
                timed_steps += 1
                if timed_steps == max(1, steps // 2):
                    sm_sample = _read_smi(vis).get("sm")  # type: ignore[assignment]
                    sm_sample = float(sm_sample) if sm_sample is not None else None
            if step == 0 or (step + 1) % 5 == 0:
                print(f"step={step + 1} loss={float(loss):.4f}", flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0 if timed_steps else 0.0
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            base.update({"ok": False, "reason": "CUDA OOM", "fit": False, "verdict": "OOM"})
            return base
        raise

    peak = torch.cuda.max_memory_allocated() / GIB
    smi = _read_smi(vis)
    tokens_done = batch * seq * timed_steps
    tok_s = (tokens_done / elapsed) if elapsed > 0 else None
    base.update(
        {
            "ok": timed_steps > 0 and not _preempt,
            "timed": timed_steps > 0,
            "steps": timed_steps,
            "warmup": warmup,
            "elapsed_s": round(elapsed, 4) if timed_steps else None,
            "tokens_per_sec": round(tok_s, 1) if tok_s is not None else None,
            "tokens_ingested": tokens_done,
            "vram_used_gb": round(peak, 2),
            "sm": sm_sample if sm_sample is not None else smi.get("sm"),
            "smi_vram_used_gb": smi.get("vram_used_gb"),
            "dtype": str(dtype).replace("torch.", ""),
            "preempt": _preempt,
            "reason": "preempt" if _preempt else ("shutdown" if _shutdown else "ok"),
        }
    )
    del opt
    del model
    torch.cuda.empty_cache()
    return base


def _print_matrix(include_nearby: bool = True, seq: int = DEFAULT_SEQ) -> None:
    sizes = list(DEFAULT_SIZES)
    if include_nearby:
        sizes.extend(NEARBY_SIZES)
    rows = estimate_matrix(sizes=sizes, seq=seq)
    print(f"# estimator  seq={seq}  Adam+bf16  reserve={RESERVE_GIB:g}GiB", flush=True)
    print(format_estimate_table(rows), flush=True)
    print("ESTIMATE_JSON " + json.dumps([asdict(r) for r in rows]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Token-ingestion throughput probe (same-node, 1 GPU)."
    )
    parser.add_argument(
        "--params", default="350M", help="Catalog size: 350M, 500M, 777M, 1B, 1.5B, 2B"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--gpu-gb", type=float, default=0.0, help="Override estimator GPU GiB (0 = detect)"
    )
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Print the full size x batch x {40,80} estimator table",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGUSR1, _handle_preempt)
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    single_estimate = any(flag in sys.argv for flag in ("--params", "--batch", "--gpu-gb"))
    if args.matrix or (args.estimate_only and not single_estimate):
        _print_matrix(seq=args.seq)
        return

    arch = parse_size(args.params)
    gpu_gb = args.gpu_gb if args.gpu_gb > 0 else 40.0
    estimate = estimate_fit(arch, batch=args.batch, seq=args.seq, gpu_gb=gpu_gb)
    if args.estimate_only:
        print(json.dumps(asdict(estimate), indent=2), flush=True)
        print(f"verdict={estimate.verdict} total_gib={estimate.total_gib}", flush=True)
        return

    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": "torch not installed; estimator-only on this host",
                    "estimate": asdict(estimate),
                    "tokens_per_sec": None,
                    "timed": False,
                }
            ),
            flush=True,
        )
        sys.exit(0)

    import torch

    if not torch.cuda.is_available():
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": "no CUDA; estimator-only",
                    "estimate": asdict(estimate),
                    "tokens_per_sec": None,
                    "timed": False,
                }
            ),
            flush=True,
        )
        sys.exit(0)

    detected = torch.cuda.get_device_properties(0).total_memory / GIB
    result = run_probe(
        arch,
        batch=args.batch,
        seq=args.seq,
        steps=args.steps,
        warmup=args.warmup,
        gpu_gb=args.gpu_gb if args.gpu_gb > 0 else detected,
    )
    print("THROUGHPUT_RESULT " + json.dumps(result), flush=True)
    endpoint = os.environ.get("PLATFORM_ENDPOINT", "")
    job_id = os.environ.get("JOB_ID", "job-local")
    run_id = os.environ.get("RUN_ID", "run-local")
    if result.get("tokens_per_sec") is not None:
        _post(
            endpoint,
            "/v1/internal/metrics",
            {
                "job_id": job_id,
                "run_id": run_id,
                "step": int(result.get("steps") or 0),
                "throughput": result["tokens_per_sec"],
                "tokens_delta": args.batch * args.seq,
                "tokens_ingested": result.get("tokens_ingested"),
                "tokens_per_sec": result["tokens_per_sec"],
            },
        )
    if _preempt:
        sys.exit(75)
    sys.exit(0 if result.get("ok") or result.get("verdict") == "NO-FIT" else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
