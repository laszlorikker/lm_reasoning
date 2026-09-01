#!/usr/bin/env python
"""M1.1 gate b: per-rule audit of the minimal-pair generator.

For a seeded sample of English train sources, regenerate minimal pairs with
rule attribution and report per rule: count, chrF(x, x⁻) mean / p10 / min
(minimal pairs should be SURFACE-CLOSE — high chrF — while flipping one
proposition), plus 20 (x, x⁻) samples per rule for eyeballing.

Outputs: runs/m1/minimal_pair_audit.json, data/fixtures/minimal_pair_samples.md
(committed), summary table for DATA.md printed.

Usage: python scripts/audit_minimal_pairs.py [--n 2000] [--corpus data/processed/pilot_v1.1/full]
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abstractnet.data.minimal_pairs import generate_minimal_pairs_bulk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/pilot_v1.1/full")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args()

    import sacrebleu
    from datasets import load_from_disk

    ds = load_from_disk(args.corpus)
    en = ds.filter(lambda r: r["lang_src"] == "en")
    sample = en.shuffle(seed=args.seed).select(range(min(args.n, len(en))))
    texts = list(sample["source_text"])

    results = generate_minimal_pairs_bulk(texts, n=4)
    per_rule: dict[str, list] = defaultdict(list)
    for x, mps in zip(texts, results):
        for m in mps:
            chrf = sacrebleu.sentence_chrf(m.text, [x]).score
            per_rule[m.rule].append((chrf, x, m.text, m.sentence_idx))

    audit, md = {}, ["# Minimal-pair audit — 20 samples per rule\n",
                    f"Seeded sample of {len(texts)} en sources from {args.corpus}; "
                    "chrF(x⁻ vs x) — high = surface-close (good minimal pair).\n"]
    print(f"{'rule':<14} {'n':>6} {'chrF mean':>10} {'p10':>7} {'min':>7}")
    for rule in sorted(per_rule):
        rows = sorted(per_rule[rule], key=lambda t: -t[0])
        scores = [r[0] for r in rows]
        stats = {
            "n": len(scores),
            "chrf_mean": round(statistics.mean(scores), 2),
            "chrf_p10": round(statistics.quantiles(scores, n=10)[0], 2) if len(scores) >= 10 else None,
            "chrf_min": round(min(scores), 2),
        }
        audit[rule] = stats
        print(f"{rule:<14} {stats['n']:>6} {stats['chrf_mean']:>10} {stats['chrf_p10']:>7} {stats['chrf_min']:>7}")
        md.append(f"\n## {rule} (n={stats['n']}, chrF mean {stats['chrf_mean']}, "
                  f"p10 {stats['chrf_p10']}, min {stats['chrf_min']})\n")
        step = max(1, len(rows) // 20)
        for chrf, x, neg, si in rows[::step][:20]:  # spread across the chrF range
            md.append(f"- chrF {chrf:.1f} (sent {si})\n  - x: {x[:300]}\n  - x⁻: {neg[:300]}")

    Path("runs/m1").mkdir(parents=True, exist_ok=True)
    Path("runs/m1/minimal_pair_audit.json").write_text(json.dumps(audit, indent=2))
    Path("data/fixtures/minimal_pair_samples.md").write_text("\n".join(md))
    print("written: runs/m1/minimal_pair_audit.json, data/fixtures/minimal_pair_samples.md")
    print("AUDIT COMPLETE")


if __name__ == "__main__":
    main()
