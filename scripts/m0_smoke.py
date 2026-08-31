#!/usr/bin/env python
"""M0 S3+S4: verify the Qwen3-1.7B-Base config against PHASE1_PLAN §2, load
fp16+SDPA, run a forward smoke test.

Usage:
    python scripts/m0_smoke.py [--config configs/base.yaml]

Hard-asserts the architecture facts the Phase-1 design depends on (28 layers so
SPLIT=14 is the middle; hidden 2048; 16 Q / 8 KV heads; tied embeddings) and
exits non-zero on mismatch. Everything else is printed and recorded.
"""

import argparse
import inspect
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abstractnet.config import load_config
from abstractnet.utils.load import load_base_lm
from abstractnet.utils import memory as mem


def verify(name: str, actual, expected) -> None:
    ok = actual == expected
    print(f"[{'OK' if ok else 'MISMATCH'}] {name}: {actual}" + ("" if ok else f" (plan expected {expected})"))
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    import transformers
    from transformers import AutoConfig

    print(f"transformers {transformers.__version__}")
    mc = AutoConfig.from_pretrained(cfg.model.name_or_path)
    print(f"model_type={mc.model_type}")

    # PHASE1_PLAN §2: "verify all of this from config.json; do not trust these numbers"
    verify("num_hidden_layers", mc.num_hidden_layers, 28)
    verify("hidden_size", mc.hidden_size, 2048)
    verify("num_attention_heads", mc.num_attention_heads, 16)
    verify("num_key_value_heads", mc.num_key_value_heads, 8)
    verify("tie_word_embeddings", mc.tie_word_embeddings, True)
    print(f"[rec] vocab_size={mc.vocab_size}  head_dim={getattr(mc, 'head_dim', None)}  "
          f"intermediate_size={mc.intermediate_size}")
    rope = next(
        (getattr(mc, a) for a in ("rope_parameters", "rope_scaling", "rope_theta") if hasattr(mc, a)),
        "n/a",
    )  # transformers v5 moved rope_theta into rope_parameters
    print(f"[rec] max_position_embeddings={mc.max_position_embeddings}  rope={rope}")
    if cfg.model.split_layer * 2 != mc.num_hidden_layers:
        print(f"[note] split_layer={cfg.model.split_layer} is not the exact middle of "
              f"{mc.num_hidden_layers} layers (fine for ablations)")

    model, tok = load_base_lm(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[rec] parameters: {n_params / 1e9:.3f} B   resident after load: {mem.resident_gib():.2f} GiB")

    attn_impl = getattr(model.config, "_attn_implementation", None) or getattr(
        model.config, "attn_implementation", None
    )
    verify("attn_implementation", attn_impl, "sdpa")
    verify("model dtype", model.dtype, torch.float16)
    verify(
        "lm_head tied to embed_tokens (shared storage)",
        model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr(),
        True,
    )

    norm_src = inspect.getsource(type(model.model.norm).forward)
    verify("RMSNorm upcasts to fp32 internally", "float32" in norm_src, True)

    B, S = cfg.profile.batch_size, cfg.profile.seq_len
    ids = torch.randint(0, mc.vocab_size, (B, S), device="cuda")
    mem.reset_peak()
    with torch.no_grad():
        out = model(ids, use_cache=False)
    torch.cuda.synchronize()
    alloc, reserved = mem.peak_gib()  # read BEFORE the checks below touch full logits
    verify("forward logits finite", bool(out.logits.isfinite().all()), True)
    print(f"[rec] logits shape {tuple(out.logits.shape)}  dtype {out.logits.dtype}  "
          f"max |logit| = {out.logits.abs().max().item():.1f}")
    print(f"[rec] forward B={B} S={S}: peak alloc {alloc:.2f} GiB, reserved {reserved:.2f} GiB (pure forward)")

    with torch.no_grad():
        for _ in range(3):  # warmup
            model(ids, use_cache=False)
        torch.cuda.synchronize()
        t0 = time.monotonic()
        iters = 10
        for _ in range(iters):
            model(ids, use_cache=False)
        torch.cuda.synchronize()
    dt = time.monotonic() - t0
    print(f"[rec] quick throughput (burst, {iters} iters): {iters * B * S / dt:,.0f} tokens/s forward")

    prompt = "The capital of France is"
    enc = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
    print(f"[rec] greedy continuation: {tok.decode(gen[0], skip_special_tokens=True)!r}")

    used, total = mem.driver_used_total_gib()
    print(f"[rec] driver-view used/total: {used:.2f} / {total:.2f} GiB")
    print("M0 SMOKE PASSED")


if __name__ == "__main__":
    main()
