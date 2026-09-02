#!/usr/bin/env python
"""M3.1 guard d: 200-step A/B — packed vs unpacked, same seed, corpus v1.2.

Isolates COMPOSITION: the packed arm's token budget is set to the unpacked
arm's tokens-per-step (effective batch 64 x mean pair length), so both arms see
the same tokens per optimizer step and per-step loss curves are directly
comparable. The only delta is length bucketing, variable pair counts, and the
contrastive negative band. (The pilot's 40k budget is throughput scaling on
top; its warmup rescale is applied when the default flips.)

Verdict: over the last 50 steps, mean |recon_packed - recon_unpacked| must be
<= 2x the unpacked rolling std (window 20). Curves + verdict -> runs/m3_1/.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PY = sys.executable
OUT = Path("runs/m3_1")
STEPS = 200
SEED = 4242


def write_cfg(name: str, packed: bool, token_budget: int) -> str:
    cfg = yaml.safe_load(Path("configs/base.yaml").read_text())
    cfg["data"]["corpus_path"] = "data/processed/pilot_v1.2/full"
    t = cfg["train"]
    t.update(seed=SEED, log_interval=25, eval_interval_steps=10**6,
             mini_report_until=0, checkpoint_interval_steps=10**6,
             checkpoint_interval_minutes=10**6.0)
    t["packing_enabled"] = packed
    if packed:
        t["token_budget"] = token_budget
        t["micro_token_budget"] = max(1500, token_budget // 3)
    path = OUT / f"ab_{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def run_arm(name: str, cfg_path: str) -> dict[int, dict]:
    run = OUT / f"ab_{name}"
    subprocess.run(["rm", "-rf", str(run)], check=True)
    r = subprocess.run([PY, "-m", "abstractnet.train", "--config", cfg_path,
                        "--run", str(run), "--steps", str(STEPS)],
                       capture_output=True, text=True, timeout=7200)
    assert r.returncode == 0, f"{name} arm failed:\n{r.stdout[-3000:]}"
    out = {}
    with open(run / "steps.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["step"]] = rec
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from datasets import load_from_disk

    ds = load_from_disk("data/processed/pilot_v1.2/full")
    mean_pair = (np.mean(ds["n_src_tokens"]) + np.mean(ds["n_tgt_tokens"]))
    budget = int(round(mean_pair * 64))
    print(f"[ab] unpacked tokens/step ≈ {budget} -> packed token_budget {budget}")

    a = run_arm("unpacked", write_cfg("unpacked", False, budget))
    print("[ab] unpacked done")
    b = run_arm("packed", write_cfg("packed", True, budget))
    print("[ab] packed done")

    steps = sorted(set(a) & set(b))
    ra = np.array([a[s]["recon"] for s in steps])
    rb = np.array([b[s]["recon"] for s in steps])
    ta = np.cumsum([a[s]["tok"] for s in steps])
    window = 20
    rolling_std = np.array([ra[max(0, i - window):i + 1].std() for i in range(len(ra))])
    last = slice(len(steps) - 50, len(steps))
    mean_abs_dev = float(np.abs(ra[last] - rb[last]).mean())
    noise = float(2 * rolling_std[last].mean())
    verdict = mean_abs_dev <= noise

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(steps, ra, label="unpacked")
    ax[0].plot(steps, rb, label="packed")
    ax[0].set_xlabel("optimizer step"); ax[0].set_ylabel("recon"); ax[0].legend()
    ax[0].set_title(f"A/B recon vs step (same tokens/step ≈ {budget})")
    ax[1].plot(ta, ra, label="unpacked")
    ax[1].plot(np.cumsum([b[s]["tok"] for s in steps]), rb, label="packed")
    ax[1].set_xlabel("cumulative pair tokens"); ax[1].legend()
    ax[1].set_title("A/B recon vs tokens")
    fig.savefig(OUT / "ab_packing.png", dpi=110, bbox_inches="tight")

    tok_a = int(np.mean([a[s]["tok"] for s in steps]))
    tok_b = int(np.mean([b[s]["tok"] for s in steps]))
    dt_a = float(np.mean([a[s]["dt_s"] for s in steps]))
    dt_b = float(np.mean([b[s]["dt_s"] for s in steps]))
    result = {
        "steps": STEPS, "token_budget": budget,
        "mean_abs_recon_dev_last50": round(mean_abs_dev, 4),
        "noise_bound_2sigma": round(noise, 4), "within_noise": verdict,
        "recon_final": {"unpacked": float(ra[-1]), "packed": float(rb[-1])},
        "tokens_per_step": {"unpacked": tok_a, "packed": tok_b},
        "s_per_step": {"unpacked": round(dt_a, 2), "packed": round(dt_b, 2)},
        "pair_tok_s": {"unpacked": round(tok_a / dt_a), "packed": round(tok_b / dt_b)},
    }
    (OUT / "ab_packing.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("AB GATE " + ("PASSED" if verdict else "FAILED"))
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
