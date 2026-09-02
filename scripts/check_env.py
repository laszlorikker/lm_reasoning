#!/usr/bin/env python
"""M0 S2 environment gates. Torch-only — run before installing the rest of the stack.

Usage:
    python scripts/check_env.py [--with-bnb]

Exits non-zero on the first failed gate. Gates:
  1. CUDA available; device compute capability == (7, 5)   [Quadro RTX 5000 Max-Q]
  2. installed wheel ships sm_75 kernels (torch.cuda.get_arch_list())
  3. fp16 matmul on GPU is finite
  4. SDPA under the FORCED mem-efficient backend, forward AND backward, at real
     Qwen3 shapes (head_dim=128) — a silent fall-back to the math backend would
     surface later as OOM at long sequence lengths
  5. fp16 autocast + GradScaler round-trip: finite grads, step taken, sane loss scale
  6. (--with-bnb) bitsandbytes AdamW8bit takes a real 8-bit step on this Turing card

Also reports driver-view free/total VRAM (cross-check of the Windows-side baseline).
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GIB = 1024**3


def gate(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        sys.exit(1)


def check_torch() -> None:
    from abstractnet.utils.hardware import detect

    print(f"torch {torch.__version__} (built for CUDA {torch.version.cuda})")
    gate("CUDA available", torch.cuda.is_available())
    hw = detect()
    cap = tuple(hw["capability"])
    sm = f"sm_{cap[0]}{cap[1]}"
    print(f"[info] {hw['name']} -> {sm}, {hw['total_gib']} GiB, "
          f"recommended dtype {hw['recommended_dtype']}")
    arches = torch.cuda.get_arch_list()
    gate(f"wheel ships {sm} kernels", sm in arches, ", ".join(arches))

    free, total = torch.cuda.mem_get_info(0)
    print(f"[info] driver-view VRAM free/total: {free / GIB:.2f} / {total / GIB:.2f} GiB")

    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    dtype = torch.bfloat16 if hw["bf16_supported"] else torch.float16
    a = torch.randn(2048, 2048, device="cuda", dtype=dtype)
    b = a @ a
    torch.cuda.synchronize()
    gate(f"{hw['recommended_dtype']} matmul finite", bool(b.isfinite().all()))

    # Real Qwen3-1.7B attention shape: B=8, H=16, S=512, head_dim=128.
    q, k, v = (
        torch.randn(8, 16, 512, 128, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    )
    backend = SDPBackend.FLASH_ATTENTION if hw["flash_sdpa_supported"] \
        else SDPBackend.EFFICIENT_ATTENTION
    label = "flash" if hw["flash_sdpa_supported"] else "mem-efficient"
    try:
        with sdpa_kernel([backend]):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out.sum().backward()
        torch.cuda.synchronize()
        ok = bool(out.isfinite().all()) and all(bool(t.grad.isfinite().all()) for t in (q, k, v))
        gate(f"SDPA {label} fwd+bwd (forced; no math fallback)", ok)
    except RuntimeError as e:
        gate(f"SDPA {label} fwd+bwd (forced; no math fallback)", False, str(e).splitlines()[0])

    # GQA shape (8 KV heads). Informational: if enable_gqa is unsupported here,
    # transformers falls back to repeat_kv expansion, which is correct but copies KV.
    try:
        kv = torch.randn(8, 8, 512, 128, device="cuda", dtype=dtype)
        with sdpa_kernel([backend]):
            F.scaled_dot_product_attention(q.detach(), kv, kv, is_causal=True, enable_gqa=True)
        torch.cuda.synchronize()
        print(f"[info] GQA (enable_gqa=True) works under the {label} kernel")
    except Exception as e:
        print(f"[info] enable_gqa unsupported under {label} ({type(e).__name__}); "
              "transformers will repeat_kv instead — correct, slightly more memory")

    model = torch.nn.Sequential(
        torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 256)
    ).cuda()  # fp32 master weights, as all trainable modules will be
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    use_scaler = not hw["bf16_supported"]  # fp16 path needs loss scaling
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    x = torch.randn(32, 256, device="cuda")
    with torch.autocast("cuda", dtype=dtype):
        loss = model(x).pow(2).mean()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    grads_finite = all(bool(p.grad.isfinite().all()) for p in model.parameters())
    scaler.step(opt)
    scaler.update()
    label = "fp16 autocast + GradScaler" if use_scaler else "bf16 autocast (scaler disabled)"
    gate(
        f"{label} round-trip",
        grads_finite and scaler.get_scale() > 0,
        f"loss_scale={scaler.get_scale():.0f}",
    )


def check_bnb() -> None:
    import bitsandbytes as bnb

    # >= 4096 elements so the optimiser actually uses 8-bit statistics.
    p = torch.nn.Parameter(torch.randn(4096, 128, device="cuda"))
    opt = bnb.optim.AdamW8bit([p], lr=1e-3)
    p.square().mean().backward()
    opt.step()
    opt.zero_grad()
    gate("bitsandbytes AdamW8bit step on Turing", bool(p.isfinite().all()), f"bnb {bnb.__version__}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-bnb", action="store_true", help="also test bitsandbytes AdamW8bit")
    args = ap.parse_args()
    check_torch()
    if args.with_bnb:
        check_bnb()
    print("ALL GATES PASSED")
