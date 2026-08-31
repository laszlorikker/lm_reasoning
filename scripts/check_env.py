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

import torch

GIB = 1024**3


def gate(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        sys.exit(1)


def check_torch() -> None:
    print(f"torch {torch.__version__} (built for CUDA {torch.version.cuda})")
    gate("CUDA available", torch.cuda.is_available())
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    gate("compute capability == (7, 5)", cap == (7, 5), f"{name} -> sm_{cap[0]}{cap[1]}")
    arches = torch.cuda.get_arch_list()
    gate("wheel ships sm_75 kernels", "sm_75" in arches, ", ".join(arches))

    free, total = torch.cuda.mem_get_info(0)
    print(f"[info] driver-view VRAM free/total: {free / GIB:.2f} / {total / GIB:.2f} GiB "
          "(includes anything Windows holds)")

    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b = a @ a
    torch.cuda.synchronize()
    gate("fp16 matmul finite", bool(b.isfinite().all()))

    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # Real Qwen3-1.7B attention shape: B=8, H=16, S=512, head_dim=128.
    q, k, v = (
        torch.randn(8, 16, 512, 128, device="cuda", dtype=torch.float16, requires_grad=True)
        for _ in range(3)
    )
    try:
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out.sum().backward()
        torch.cuda.synchronize()
        ok = bool(out.isfinite().all()) and all(bool(t.grad.isfinite().all()) for t in (q, k, v))
        gate("SDPA mem-efficient fwd+bwd (forced; no math fallback)", ok)
    except RuntimeError as e:
        gate("SDPA mem-efficient fwd+bwd (forced; no math fallback)", False, str(e).splitlines()[0])

    # GQA shape (8 KV heads). Informational: if enable_gqa is unsupported here,
    # transformers falls back to repeat_kv expansion, which is correct but copies KV.
    try:
        kv = torch.randn(8, 8, 512, 128, device="cuda", dtype=torch.float16)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            F.scaled_dot_product_attention(q.detach(), kv, kv, is_causal=True, enable_gqa=True)
        torch.cuda.synchronize()
        print("[info] GQA (enable_gqa=True) works under the mem-efficient kernel")
    except Exception as e:
        print(f"[info] enable_gqa unsupported under mem-efficient ({type(e).__name__}); "
              "transformers will repeat_kv instead — correct, slightly more memory")

    model = torch.nn.Sequential(
        torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 256)
    ).cuda()  # fp32 master weights, as all trainable modules will be
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(32, 256, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(x).pow(2).mean()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    grads_finite = all(bool(p.grad.isfinite().all()) for p in model.parameters())
    scaler.step(opt)
    scaler.update()
    gate(
        "fp16 autocast + GradScaler round-trip",
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
