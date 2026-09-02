#!/usr/bin/env python
"""Diagnose model-invisible hard negatives (pilot step-500 finding).

A negative whose perturbed sentence lies beyond the 8-chunk/512-token cap is
byte-identical to its source in model-visible input_ids: cos=1.000 by
construction — an impossible negative.

Counts, broken down by origin / kind / rule / K:
  TRAIN (v1.2): stored src_ids vs stored neg_ids (both already capped; the
  pack path dropped overflowing negatives, so the prediction is ~0 — verify).
  POOL (v2) and PANEL: re-tokenize text and negatives with the encode-time
  caps and compare.
Also extracts the ten failure-table rows from the pilot's step-500 report and
confirms membership. CPU-only; does not touch the running pilot.

Usage: python scripts/audit_identical_negatives.py [--report <report.html>]
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abstractnet.config import load_config
from abstractnet.data.chunking import chunk_document


def audit_train(cfg, out: dict) -> None:
    from datasets import load_from_disk

    ds = load_from_disk(cfg.data.corpus_path)
    ident = Counter()
    total = Counter()
    n_ident = n_total = 0
    for r in ds:
        if not r["neg_ids"]:
            continue
        for n in r["neg_ids"]:
            n_total += 1
            total[r["origin"].split("-")[0]] += 1
            if list(n) == list(r["src_ids"]):
                n_ident += 1
                ident[f"{r['origin'].split('-')[0]}|K={r['k']}"] += 1
    out["train"] = {"negatives_total": n_total, "identical": n_ident,
                    "identical_by_origin_k": dict(ident),
                    "negatives_by_origin": dict(total)}


def audit_fixture(path: str, tok, cfg, out: dict, key: str) -> set[str]:
    dcfg = cfg.data
    affected_ids: set[str] = set()
    ident = Counter()
    trunc_docs = 0
    n_total = n_ident = 0
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        cd = chunk_document(d["text"], tok, d["lang"], dcfg.max_chunk_tokens,
                            dcfg.max_chunks, dcfg.max_source_tokens)
        if cd is None:
            continue
        if cd.n_dropped_tokens > 0:
            trunc_docs += 1
        for neg in d.get("hard_negatives", []):
            n_total += 1
            nd = chunk_document(neg["text"], tok, d["lang"], dcfg.max_chunk_tokens,
                                dcfg.max_chunks, dcfg.max_source_tokens)
            if nd is not None and nd.input_ids == cd.input_ids:
                n_ident += 1
                affected_ids.add(d["id"])
                ident[f"{d['origin'].split('-')[0]}|{neg['kind']}|"
                      f"rule={neg.get('rule')}|K={cd.k}"] += 1
    out[key] = {"negatives_total": n_total, "identical": n_ident,
                "docs_truncated_at_encode": trunc_docs,
                "identical_by_origin_kind_rule_k": dict(ident)}
    return affected_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--report", default=None,
                    help="step-500 report.html to cross-check the failure table")
    args = ap.parse_args()
    cfg = load_config(args.config)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    out: dict = {}
    affected = audit_fixture(cfg.data.val_pool_path, tok, cfg, out, "pool")
    audit_fixture(cfg.data.panel_path, tok, cfg, out, "panel")
    audit_train(cfg, out)

    if args.report and Path(args.report).exists():
        html = Path(args.report).read_text()
        table2 = html.split("hard negatives closest", 1)[-1]
        rows = re.findall(r"<td>(vp-\d+)</td>", table2)[:10]
        out["failure_table_rows"] = rows
        out["failure_rows_in_affected_set"] = sum(r in affected for r in rows)

    Path("runs/m3_1").mkdir(parents=True, exist_ok=True)
    Path("runs/m3_1/identical_negatives_audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("AUDIT COMPLETE")


if __name__ == "__main__":
    main()
