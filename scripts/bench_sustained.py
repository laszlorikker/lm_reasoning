#!/usr/bin/env python
"""M0 S4: sustained forward throughput under the Max-Q power/thermal cap.

Runs a forward-only loop at the profile batch/seq for --minutes, sampling
nvidia-smi (SM clock, power, temperature, memory) every --sample-s in a thread.
Reports minute-1 (burst) vs minutes 2..N (sustained) tokens/s and telemetry.
Writes runs/m0/sustained_telemetry.csv and runs/m0/sustained_summary.json.

Usage:
    python scripts/bench_sustained.py [--config configs/base.yaml] [--minutes 10]
"""

import argparse
import json
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abstractnet.config import load_config
from abstractnet.utils.load import load_base_lm
from abstractnet.utils import memory as mem


def telemetry_loop(stop: threading.Event, rows: list, sample_s: float, t0: float) -> None:
    while not stop.is_set():
        try:
            rows.append((time.monotonic() - t0, mem.nvidia_smi_sample()))
        except Exception as e:  # keep the bench alive if nvidia-smi hiccups
            rows.append((time.monotonic() - t0, f"error: {e}"))
        stop.wait(sample_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--sample-s", type=float, default=5.0)
    ap.add_argument("--out-dir", default="runs/m0")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _ = load_base_lm(cfg.model)
    B, S = cfg.profile.batch_size, cfg.profile.seq_len
    vocab = model.config.vocab_size
    ids = torch.randint(0, vocab, (B, S), device="cuda")

    with torch.no_grad():  # warmup outside the clock
        for _ in range(3):
            model(ids, use_cache=False)
        torch.cuda.synchronize()

    t0 = time.monotonic()
    stop = threading.Event()
    rows: list = []
    sampler = threading.Thread(target=telemetry_loop, args=(stop, rows, args.sample_s, t0), daemon=True)
    sampler.start()

    tokens_per_min: dict[int, int] = defaultdict(int)
    iters = 0
    with torch.no_grad():
        while (elapsed := time.monotonic() - t0) < args.minutes * 60:
            model(ids, use_cache=False)
            torch.cuda.synchronize()
            tokens_per_min[int(elapsed // 60)] += B * S
            iters += 1
    stop.set()
    sampler.join(timeout=10)

    def parse(row: str) -> tuple[float, float, float, float] | None:
        try:
            clock, power, temp, mem_used = (float(x) for x in row.split(","))
            return clock, power, temp, mem_used
        except ValueError:
            return None

    per_min_telemetry: dict[int, list] = defaultdict(list)
    with open(out_dir / "sustained_telemetry.csv", "w") as f:
        f.write("t_s,sm_clock_mhz,power_w,temp_c,mem_used_mib\n")
        for t, row in rows:
            f.write(f"{t:.1f},{row}\n")
            if (p := parse(row)) is not None:
                per_min_telemetry[int(t // 60)].append(p)

    minutes = sorted(tokens_per_min)
    complete = [m for m in minutes if m < int(args.minutes)]  # drop any partial last bucket
    per_min_tps = {m: tokens_per_min[m] / 60.0 for m in complete}
    burst = per_min_tps.get(0, 0.0)
    sustained_mins = [m for m in complete if m >= 1]
    sustained = statistics.mean(per_min_tps[m] for m in sustained_mins) if sustained_mins else None
    last = per_min_tps[complete[-1]] if complete else None

    def med(minute: int, idx: int):
        vals = [r[idx] for r in per_min_telemetry.get(minute, [])]
        return round(statistics.median(vals), 1) if vals else None

    summary = {
        "batch": B, "seq_len": S, "minutes": args.minutes, "iters": iters,
        "tokens_per_s": {
            "minute_1_burst": round(burst, 1),
            "minutes_2_to_end_sustained": round(sustained, 1) if sustained else None,
            "last_minute": round(last, 1) if last else None,
            "per_minute": {m: round(v, 1) for m, v in per_min_tps.items()},
        },
        "telemetry_medians": {
            f"minute_{m + 1}": {
                "sm_clock_mhz": med(m, 0), "power_w": med(m, 1),
                "temp_c": med(m, 2), "mem_used_mib": med(m, 3),
            }
            for m in sorted(per_min_telemetry)
        },
        "max_temp_c": max((r[2] for rs in per_min_telemetry.values() for r in rs), default=None),
    }
    (out_dir / "sustained_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== sustained bench: B={B} S={S}, {args.minutes:.0f} min, {iters} iters ===")
    print(f"{'min':>4} {'tok/s':>10} {'SM MHz':>8} {'W':>7} {'degC':>5}")
    for m in complete:
        print(f"{m + 1:>4} {per_min_tps[m]:>10,.0f} {med(m, 0) or '-':>8} {med(m, 1) or '-':>7} {med(m, 2) or '-':>5}")
    print(f"\nburst (minute 1):     {burst:>10,.0f} tokens/s")
    if sustained is not None:
        print(f"sustained (min 2..N): {sustained:>10,.0f} tokens/s")
    print(f"max temp: {summary['max_temp_c']} degC")
    print(f"written: {out_dir}/sustained_summary.json, {out_dir}/sustained_telemetry.csv")


if __name__ == "__main__":
    main()
