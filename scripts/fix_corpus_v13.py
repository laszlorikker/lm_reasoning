#!/usr/bin/env python
"""Corpus v1.3 (pilot step-500 fix): strip the 55 model-invisible negatives
(capped neg_ids == src_ids — tokenizer collisions and out-of-window
substitutions) from the affected shards; every other shard is hardlinked.
The finalize that follows re-runs the dedup gate with a hash set rebuilt to
include pool_v3. CPU-only; the running pilot keeps its v1.2 mmap untouched.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SRC = Path("data/processed/pilot_v1.2")
DST = Path("data/processed/pilot_v1.3")
AFFECTED = ["paws_x_singles", "nli_entailment", "concat_paraphrase"]


def strip_shard(name: str, counters: dict) -> None:
    from datasets import load_from_disk

    ds = load_from_disk(str(SRC / name))

    def strip(r):
        keep = [i for i, n in enumerate(r["neg_ids"]) if list(n) != list(r["src_ids"])]
        counters["stripped"] += len(r["neg_ids"]) - len(keep)
        if len(keep) == len(r["neg_ids"]):
            return {}
        return {"neg_texts": [r["neg_texts"][i] for i in keep],
                "neg_ids": [r["neg_ids"][i] for i in keep],
                "neg_spans": [r["neg_spans"][i] for i in keep]}

    ds = ds.map(strip)
    ds.save_to_disk(str(DST / name))
    shutil.copy(SRC / name / "build_stats.json", DST / name / "build_stats.json")


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for shard in sorted(SRC.iterdir()):
        if not shard.is_dir() or shard.name in AFFECTED + ["full"]:
            continue
        if not (DST / shard.name).exists():
            subprocess.run(["cp", "-al", str(shard), str(DST / shard.name)], check=True)
    counters = {"stripped": 0}
    for name in AFFECTED:
        strip_shard(name, counters)
        print(f"[v1.3] {name}: rebuilt (running strip count {counters['stripped']})")
    print(json.dumps(counters))
    print("V13 SHARDS COMPLETE")


if __name__ == "__main__":
    main()
