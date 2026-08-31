#!/usr/bin/env python
"""Memory profiler — rerun at every milestone that touches the model (kickoff rule 4).

M0 mode (--profile split_forward), all at the configured profile batch/seq:
  a. weights resident after load
  b. encoder half: embed + layers [0, split), full sequence, no grad
  c. full-stack teacher-forcing forward incl. full-vocab logits, no grad
  d. full-stack forward with the LM head applied in 1024-token chunks (§4 chunked-CE preview)
  e. b then c back-to-back — the M0 "split forward" envelope; HARD budget gate
  f. training-shaped preview: params frozen + input-grads enabled (the PEFT/M2
     pattern), fp16 autocast, chunked CE, backward; gradient checkpointing on/off.
     Informational — M2's real train-step profile is the binding number.

Exits non-zero if (e) peak reserved exceeds profile.vram_budget_gib.

Usage:
    python scripts/profile_memory.py [--config configs/base.yaml] [--profile split_forward]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abstractnet.config import load_config
from abstractnet.utils.load import load_base_lm
from abstractnet.utils import memory as mem

CHUNK_TOKENS = 1024  # §4: compute LM-head cross-entropy in chunks of this many tokens


def embed_and_layers(model, input_ids: torch.Tensor, n_layers: int) -> torch.Tensor:
    """Embed + first n_layers decoder layers, full sequence, causal, no KV cache.

    This is what M2's encoder half will do; attention_mask=None puts SDPA on the
    is_causal fast path. The pooling head does not exist yet (M2).
    """
    m = model.model
    h = m.embed_tokens(input_ids)
    pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    pos_emb = m.rotary_emb(h, pos)
    for layer in m.layers[:n_layers]:
        out = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pos_emb, use_cache=False)
        h = out[0] if isinstance(out, tuple) else out
    return h


def chunked_ce(model, hidden: torch.Tensor, targets: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
    """Cross-entropy over the full vocab, CHUNK_TOKENS *flattened* positions at a
    time (chunking the sequence axis alone is no chunking at all when B×S fits
    one slice). With use_checkpoint=True each chunk's logits are recomputed
    during backward, so only one chunk is ever alive — the configuration M2's
    training loss will use.
    """
    h = hidden.flatten(0, 1)
    t = targets.flatten()

    def chunk_loss(h_chunk: torch.Tensor, t_chunk: torch.Tensor) -> torch.Tensor:
        logits = model.lm_head(h_chunk)
        return F.cross_entropy(logits.float(), t_chunk, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], CHUNK_TOKENS):
        hc, tc = h[i : i + CHUNK_TOKENS], t[i : i + CHUNK_TOKENS]
        if use_checkpoint:
            total = total + torch.utils.checkpoint.checkpoint(chunk_loss, hc, tc, use_reentrant=False)
        else:
            total = total + chunk_loss(hc, tc)
    return total / t.numel()


def report(tag: str, note: str = "") -> tuple[float, float]:
    alloc, reserved = mem.peak_gib()
    used, total = mem.driver_used_total_gib()
    print(f"{tag:<58} peak alloc {alloc:6.2f} GiB | peak reserved {reserved:6.2f} GiB | "
          f"driver {used:5.2f}/{total:.2f} GiB{'  ' + note if note else ''}")
    return alloc, reserved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--profile", default="split_forward", choices=["split_forward"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    B, S, budget = cfg.profile.batch_size, cfg.profile.seq_len, cfg.profile.vram_budget_gib
    split = cfg.model.split_layer

    model, _ = load_base_lm(cfg.model)
    vocab = model.config.vocab_size
    ids = torch.randint(0, vocab, (B, S), device="cuda")
    targets = torch.randint(0, vocab, (B, S), device="cuda")
    print(f"=== memory profile: {args.profile}, B={B}, S={S}, split={split}, budget {budget:.1f} GiB ===")

    mem.reset_peak()
    print(f"{'a. weights resident after load':<58} {mem.resident_gib():6.2f} GiB alloc")

    with torch.no_grad():
        mem.reset_peak()
        h = embed_and_layers(model, ids, split)
        torch.cuda.synchronize()
        report(f"b. encoder half fwd (layers 0..{split - 1})")
        del h

        mem.reset_peak()
        out = model(ids, use_cache=False)
        torch.cuda.synchronize()
        report("c. full-stack fwd, full-vocab logits")
        del out

        mem.reset_peak()
        hidden = model.model(ids, use_cache=False).last_hidden_state
        loss = chunked_ce(model, hidden, targets)
        torch.cuda.synchronize()
        report(f"d. full-stack fwd, {CHUNK_TOKENS}-token chunked CE", f"loss={loss.item():.2f}")
        del hidden, loss

        mem.reset_peak()
        h = embed_and_layers(model, ids, split)
        out = model(ids, use_cache=False)
        torch.cuda.synchronize()
        e_alloc, e_reserved = report("e. SPLIT FORWARD: encoder half + full decoder stack")
        del h, out

    # f. training-shaped preview: frozen params, grads flow through activations only
    #    (exactly the PEFT + gradient-checkpointing pattern M2 will use).
    model.requires_grad_(False)
    model.enable_input_require_grads()
    model.train()
    scaler = torch.amp.GradScaler("cuda")

    def train_shaped(checkpointing: bool, note: str) -> None:
        # NB: on WSL2 the driver oversubscribes into host RAM instead of raising
        # OOM — an over-budget run completes, silently and slowly. Wall time is
        # printed so the spill shows up even when no error does.
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        mem.reset_peak()
        t0 = time.monotonic()
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                hidden = model.model(ids, use_cache=False).last_hidden_state
                loss = chunked_ce(model, hidden, targets, use_checkpoint=True)
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            report(
                f"f. train-shaped fwd+bwd, checkpointing={'on' if checkpointing else 'off'}",
                f"{note} {time.monotonic() - t0:.1f}s",
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{'f. train-shaped fwd+bwd, checkpointing=' + ('on' if checkpointing else 'off'):<58} "
                  f"OOM at B={B} — finding, not failure; M2 profile is binding")
            torch.cuda.empty_cache()

    train_shaped(checkpointing=True, note="(preview)")
    train_shaped(checkpointing=False, note="(preview)")

    print(f"\nbudget gate on (e): peak reserved {e_reserved:.2f} GiB vs budget {budget:.1f} GiB "
          f"(alloc {e_alloc:.2f} GiB)")
    if e_reserved > budget:
        print("BUDGET EXCEEDED — failing loudly (kickoff rule 4)")
        sys.exit(1)
    print("WITHIN BUDGET")


if __name__ == "__main__":
    main()
