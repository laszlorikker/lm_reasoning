#!/usr/bin/env python
"""M1 S1: verify every candidate Hub dataset exists and has the expected fields
BEFORE building on it (kickoff rule 8). Streaming probes — no bulk downloads.

Prints a table; writes full details to runs/m1/dataset_verification.json.
Substitutions and unavailability get recorded in DATA.md by hand afterwards.

Usage: python scripts/verify_datasets.py [--only KEY]
"""

import argparse
import itertools
import json
import traceback
from pathlib import Path

# (key, hub_id, config, split, fields we need)
CANDIDATES = [
    # paraphrase + adversarial negatives (label 0 in PAWS is the adversarial one)
    ("paws-x-en", "google-research-datasets/paws-x", "en", "train", ["sentence1", "sentence2", "label"]),
    ("paws-x-fr", "google-research-datasets/paws-x", "fr", "train", ["sentence1", "sentence2", "label"]),
    ("paws-x-de", "google-research-datasets/paws-x", "de", "train", ["sentence1", "sentence2", "label"]),
    ("paws-x-es", "google-research-datasets/paws-x", "es", "train", ["sentence1", "sentence2", "label"]),
    ("glue-mrpc", "nyu-mll/glue", "mrpc", "train", ["sentence1", "sentence2", "label"]),
    ("glue-qqp", "nyu-mll/glue", "qqp", "train", ["question1", "question2", "label"]),
    ("tapaco", "tapaco", "all_languages", "train", ["paraphrase_set_id", "paraphrase", "language"]),
    # translation, sentence level
    ("opus100-de-en", "Helsinki-NLP/opus-100", "de-en", "train", ["translation"]),
    ("opus100-en-es", "Helsinki-NLP/opus-100", "en-es", "train", ["translation"]),
    ("opus100-en-fr", "Helsinki-NLP/opus-100", "en-fr", "train", ["translation"]),
    ("opus100-en-it", "Helsinki-NLP/opus-100", "en-it", "train", ["translation"]),
    ("opus100-en-pt", "Helsinki-NLP/opus-100", "en-pt", "train", ["translation"]),
    ("tatoeba-en-fr", "Helsinki-NLP/tatoeba", "en-fr", "train", ["translation"]),
    ("tatoeba_mt", "Helsinki-NLP/tatoeba_mt", "eng-fra", "test", None),
    # translation, document-ordered rows (multi-chunk via consecutive concat)
    ("europarl-en-fr", "Helsinki-NLP/europarl", "en-fr", "train", ["translation"]),
    ("europarl-de-en", "Helsinki-NLP/europarl", "de-en", "train", ["translation"]),
    ("europarl-en-es", "Helsinki-NLP/europarl", "en-es", "train", ["translation"]),
    ("europarl-en-it", "Helsinki-NLP/europarl", "en-it", "train", ["translation"]),
    ("europarl-en-pt", "Helsinki-NLP/europarl", "en-pt", "train", ["translation"]),
    ("newscomm-en-fr", "Helsinki-NLP/news_commentary", "en-fr", "train", ["translation"]),
    ("newscomm-de-en", "Helsinki-NLP/news_commentary", "de-en", "train", ["translation"]),
    ("newscomm-en-es", "Helsinki-NLP/news_commentary", "en-es", "train", ["translation"]),
    # NLI: weak positives (entailment) / hard negatives (contradiction)
    ("xnli-all", "facebook/xnli", "all_languages", "validation", ["premise", "hypothesis", "label"]),
    ("multi_nli", "nyu-mll/multi_nli", None, "train", ["premise", "hypothesis", "label"]),
    ("anli", "facebook/anli", None, "train_r1", ["premise", "hypothesis", "label"]),
    # multiple derivations of one conclusion (natural multi-sentence text)
    ("openmath2", "nvidia/OpenMathInstruct-2", None, "train_1M", ["problem", "generated_solution", "problem_source"]),
    # reasoning probe (eval only)
    ("babi-15-16", "facebook/babi_qa", "en-valid-10k", "train", None),
    ("babi-alt", "Muennighoff/babi", None, "train", ["passage", "question", "answer", "task"]),
]


def probe(hub_id: str, config, split: str, want_fields):
    import datasets

    ds = datasets.load_dataset(hub_id, config, split=split, streaming=True)
    rows = list(itertools.islice(ds, 2))
    if not rows:
        return {"status": "EMPTY", "fields": [], "sample": None}
    fields = sorted(rows[0].keys())
    missing = [f for f in (want_fields or []) if f not in rows[0]]
    sample = {k: (str(v)[:120] + "…" if len(str(v)) > 120 else v) for k, v in rows[0].items()}
    return {
        "status": "OK" if not missing else f"MISSING_FIELDS:{missing}",
        "fields": fields,
        "sample": sample,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="probe a single key")
    args = ap.parse_args()

    out: dict = {}
    for key, hub_id, config, split, want in CANDIDATES:
        if args.only and key != args.only:
            continue
        try:
            out[key] = {"hub_id": hub_id, "config": config, "split": split, **probe(hub_id, config, split, want)}
        except Exception as e:
            out[key] = {
                "hub_id": hub_id, "config": config, "split": split,
                "status": f"UNAVAILABLE ({type(e).__name__})",
                "error": str(e).splitlines()[0][:200] if str(e) else traceback.format_exc().splitlines()[-1],
            }
        row = out[key]
        print(f"{key:<18} {row['status']:<28} {hub_id}" + (f" [{config}]" if config else ""))

    Path("runs/m1").mkdir(parents=True, exist_ok=True)
    Path("runs/m1/dataset_verification.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nfull details -> runs/m1/dataset_verification.json")


if __name__ == "__main__":
    main()
