#!/usr/bin/env python
"""M2: single-batch overfit — proves gradients reach every new module.

200 steps on ONE batch of 4 real pairs with the real optimiser stack:
AdamW8bit (new modules lr 2e-4, LoRA lr 1e-4 — PHASE1_PLAN §5), fp16 autocast
+ GradScaler, gradient checkpointing, grad clip 1.0, memory-fraction guard.

Hard asserts (exit non-zero): L_recon final <= 0.5 x initial; every gate moved
off zero; no non-finite loss ever; loss scale did not collapse.

Usage: python scripts/m2_overfit.py [--config configs/base.yaml] [--steps 200]
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abstractnet.config import load_config
from abstractnet.utils import memory as mem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mem.apply_budget_guard(cfg.profile.vram_budget_gib)

    import bitsandbytes as bnb

    from abstractnet.data.sampling import real_batch
    from abstractnet.modeling.abstract_lm import AbstractLM

    corpus = next(p for p in ("data/processed/pilot_v1.1/full", "data/processed/pilot_v1/full")
                  if Path(p).exists())
    model = AbstractLM(cfg)
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    batch = real_batch(corpus, args.batch, model.tokenizer, seed=3,
                       require_negatives=True, mixed_k=True)
    print(f"batch from {corpus}: types={batch['meta']['pair_type']}")

    new_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    opt = bnb.optim.AdamW8bit(
        [{"params": new_params, "lr": 2e-4}, {"params": lora_params, "lr": 1e-4}],
        weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    trainables = new_params + lora_params

    recon_hist, scale_hist = [], []
    t0 = time.monotonic()
    for step in range(args.steps):
        out = model(batch)
        loss = out["loss"]
        if not bool(torch.isfinite(loss)):
            print(f"FAIL: non-finite loss at step {step}")
            sys.exit(1)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainables, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        recon_hist.append(float(out["recon"]))
        scale_hist.append(scaler.get_scale())
        if step % 25 == 0 or step == args.steps - 1:
            g = out["gates"]
            print(f"step {step:>3}  recon {recon_hist[-1]:.4f}  con {float(out['contrastive']):.4f}  "
                  f"|gate| max {g.abs().max():.4f} mean {g.abs().mean():.4f}  scale {scale_hist[-1]:.0f}")

    dt = (time.monotonic() - t0) / args.steps
    alloc, reserved = mem.peak_gib()
    print(f"\n{dt:.2f} s/step | peak alloc {alloc:.2f} GiB reserved {reserved:.2f} GiB")

    first, last = recon_hist[0], recon_hist[-1]
    gates = out["gates"].abs()
    ok = True
    if last > 0.5 * first:
        print(f"FAIL: recon {first:.3f} -> {last:.3f}, less than 2x drop")
        ok = False
    if not bool((gates > 0).all()):
        print(f"FAIL: some gates never moved: {gates.tolist()}")
        ok = False
    if scale_hist[-1] < 128:
        print(f"FAIL: loss scale collapsed to {scale_hist[-1]}")
        ok = False
    print("OVERFIT " + ("PASSED" if ok else "FAILED") +
          f": recon {first:.3f} -> {last:.3f} ({last / first:.1%}), "
          f"gates |max| {gates.max():.4f}, scale {scale_hist[-1]:.0f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
