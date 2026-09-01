"""Eval-leakage dedup (M1.1 gate a): every eval source held out by content hash.

Hash = sha1 of whitespace-normalized text. The eval hash set covers the frozen
fixtures (pool text, paraphrase, translations, negatives; panel likewise) and
the full held-out splits our evals draw from, for the six pilot languages:
paws-x test (en/fr/de/es), xnli validation+test (en/fr/de/es sides), opus-100
test (or validation) both sides, multi_nli validation_matched, glue qqp
validation, glue mrpc validation+test.

The pilot finalize drops any train pair whose source or target hash collides,
and strips colliding negatives; the removed counts go to DATA.md.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

CACHE = Path("runs/m1/eval_hashes.txt")


def norm_hash(text: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", text.strip()).encode()).hexdigest()


def _fixture_texts(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        d = json.loads(line)
        yield d["text"]
        if d.get("paraphrase"):
            yield d["paraphrase"]
        yield from d.get("translations", {}).values()
        for n in d.get("hard_negatives", []):
            yield n["text"]


def build_eval_hash_set(fixtures_dir: str | Path = "data/fixtures") -> set[str]:
    import datasets

    from abstractnet.data.pairs import OPUS_CFGS

    H: set[str] = set()

    def add(*texts):
        for t in texts:
            if t and t.strip():
                H.add(norm_hash(t))

    for name in ("val_pool_v1.jsonl", "val_pool_v2.jsonl", "panel_v1.jsonl"):
        for t in _fixture_texts(Path(fixtures_dir) / name):
            add(t)

    for lang in ("en", "fr", "de", "es"):
        for r in datasets.load_dataset("google-research-datasets/paws-x", lang, split="test"):
            add(r["sentence1"], r["sentence2"])
    for split in ("validation", "test"):
        for r in datasets.load_dataset("facebook/xnli", "all_languages", split=split):
            hyp = dict(zip(r["hypothesis"]["language"], r["hypothesis"]["translation"]))
            for lang in ("en", "fr", "de", "es"):
                add(r["premise"].get(lang), hyp.get(lang))
    for cfg in OPUS_CFGS:
        a, b = cfg.split("-")
        try:
            ds = datasets.load_dataset("Helsinki-NLP/opus-100", cfg, split="test")
        except Exception:
            ds = datasets.load_dataset("Helsinki-NLP/opus-100", cfg, split="validation")
        for r in ds:
            add(r["translation"][a], r["translation"][b])
    for r in datasets.load_dataset("nyu-mll/multi_nli", split="validation_matched"):
        add(r["premise"], r["hypothesis"])
    for r in datasets.load_dataset("nyu-mll/glue", "qqp", split="validation"):
        add(r["question1"], r["question2"])
    for split in ("validation", "test"):
        for r in datasets.load_dataset("nyu-mll/glue", "mrpc", split=split):
            add(r["sentence1"], r["sentence2"])
    return H


def load_or_build_eval_hashes() -> set[str]:
    if CACHE.exists():
        return set(CACHE.read_text().split())
    H = build_eval_hash_set()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text("\n".join(sorted(H)))
    return H
