#!/usr/bin/env python
"""M3 item 2: sweep micro-batch ∈ {8, 12, 16} at worst-case document shape.

Reports peak memory and pair-tokens/s per size; recommends the fastest that
stays under budget with >= 1.5 GiB slack. Effective batch stays 64 through
accumulation regardless. Result table -> runs/m3/batch_sweep.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abstractnet.config import load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_memory import profile_train_step  # noqa: E402


def main() -> None:
    cfg = load_config("configs/base.yaml")
    budget = cfg.profile.vram_budget_gib
    results = []
    for micro in (8, 12, 16):
        if 64 % micro:
            print(f"[sweep] micro={micro}: 64 not divisible -> accumulation would drift; "
                  "measured anyway for the table")
        r = profile_train_step(cfg, micro=micro, gate=False)
        results.append(r)
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()

    ok = [r for r in results if not r.get("oom") and budget - r["reserved_gib"] >= 1.5
          and 64 % r["micro"] == 0]
    best = max(ok, key=lambda r: r["pair_tok_s"]) if ok else None
    print(f"\n{'micro':>6} {'reserved GiB':>13} {'s/step':>7} {'pair tok/s':>11} {'accum(64)':>10}")
    for r in results:
        if r.get("oom"):
            print(f"{r['micro']:>6} {'OOM':>13}")
            continue
        accum = "64/" + str(r["micro"]) if 64 % r["micro"] else "n/a"
        print(f"{r['micro']:>6} {r['reserved_gib']:>13} {r['s_per_step']:>7} "
              f"{r['pair_tok_s']:>11,} {accum:>10}")
    Path("runs/m3").mkdir(parents=True, exist_ok=True)
    Path("runs/m3/batch_sweep.json").write_text(json.dumps(
        {"results": results, "budget_gib": budget,
         "recommended_micro": best["micro"] if best else None}, indent=2))
    print(f"\nrecommended micro-batch: {best['micro'] if best else 'none under budget+slack'}")
    print("SWEEP COMPLETE")


if __name__ == "__main__":
    main()
