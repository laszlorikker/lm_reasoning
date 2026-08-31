#!/usr/bin/env python
"""M1 S4: build the pilot pair corpus (~50M source+target tokens, PHASE1_PLAN §3).

Idempotent per source: shards live in data/processed/pilot_v1/<source>/ and an
existing shard is skipped (delete its dir to rebuild). The final dataset is
data/processed/pilot_v1/full (concatenated, shuffled). Stats — counts, token
sums, achieved-K histogram, language mix, drop/fix counters — go to
runs/m1/pilot_stats.json; DATA.md is written from those numbers.

Usage:
    python scripts/build_pilot.py [--config configs/base.yaml] [--source KEY | --finalize]
    (no args: build every missing source, then finalize)
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abstractnet.config import load_config
from abstractnet.data.chunking import chunk_document
from abstractnet.data.minimal_pairs import generate_minimal_pairs_bulk, hash_seed
from abstractnet.data import pairs as P

OUT = Path("data/processed/pilot_v1")
FEATURES = None  # set lazily (datasets import is slow)


def features():
    global FEATURES
    if FEATURES is None:
        from datasets import Features, Sequence, Value

        FEATURES = Features({
            "id": Value("string"),
            "source_text": Value("string"), "target_text": Value("string"),
            "lang_src": Value("string"), "lang_tgt": Value("string"),
            "pair_type": Value("string"), "origin": Value("string"),
            "src_ids": Sequence(Value("int32")), "src_spans": Sequence(Value("int32")),
            "tgt_ids": Sequence(Value("int32")), "tgt_spans": Sequence(Value("int32")),
            "k": Value("int32"), "tgt_k": Value("int32"),
            "n_src_tokens": Value("int32"), "n_tgt_tokens": Value("int32"),
            "neg_texts": Sequence(Value("string")),
            "neg_ids": Sequence(Sequence(Value("int32"))),
            "neg_spans": Sequence(Sequence(Value("int32"))),
        })
    return FEATURES


def extract(source: str, pcfg: dict, stats: Counter):
    seed = pcfg["seed"]
    s = pcfg["sources"][source]
    if source == "opus100_singles":
        yield from P.iter_opus100_singles(s["n"], seed)
    elif source == "concat_translation":
        yield from P.iter_concat_translation(s["n"], seed, s["k_min"], s["k_max"], stats)
    elif source == "paws_x_singles":
        pools = P.load_paws_pools(s["langs"])
        for lang in s["langs"]:
            yield from P.iter_paraphrase_singles(pools[lang][: s["per_lang"]], lang, f"paws-x-{lang}", s["per_lang"], seed)
    elif source == "concat_paraphrase":
        paws_cfg = pcfg["sources"]["paws_x_singles"]
        qqp_used = pcfg["sources"]["qqp_singles"]["n"]
        pools = P.load_paws_pools([l for l in s["lang_shares"] if l != "en"])
        rest = {lang: pool[paws_cfg["per_lang"]:] for lang, pool in pools.items()}
        rest["en"] = P.load_qqp_pool()[qqp_used:]
        yield from P.iter_concat_paraphrase(rest, s["lang_shares"], s["n"], seed, s["k_min"], s["k_max"])
    elif source == "qqp_singles":
        yield from P.iter_paraphrase_singles(P.load_qqp_pool()[: s["n"]], "en", "glue-qqp", s["n"], seed)
    elif source == "mrpc_singles":
        yield from P.iter_paraphrase_singles(P.load_mrpc_pool()[: s["n"]], "en", "glue-mrpc", s["n"], seed)
    elif source == "nli_entailment":
        yield from P.iter_nli_entailment(s["n"], seed)
    elif source == "math_derivation":
        yield from P.iter_math_derivations(s["n"], seed)
    else:
        raise KeyError(f"unknown source {source!r}")


def add_generated_negatives(examples: list, pcfg: dict, stats: Counter) -> None:
    share = pcfg["negatives"]["minimal_pair_share_en"]
    per = pcfg["negatives"]["per_example"]
    rng = random.Random(pcfg["seed"] ^ 0xBEEF)
    idx = [i for i, ex in enumerate(examples) if ex.lang_src == "en" and rng.random() < share]
    texts = [examples[i].source for i in idx]
    t0 = time.monotonic()
    results = generate_minimal_pairs_bulk(texts, n=per)
    for i, mps in zip(idx, results):
        examples[i].negatives += [m.text for m in mps]
        stats["neg_generated"] += len(mps)
    stats["neg_gen_seconds"] += int(time.monotonic() - t0)


def pack(examples: list, tok, dcfg, stats: Counter, k_hist: Counter, langs: Counter) -> list[dict]:
    rows = []
    for ex in examples:
        doc = chunk_document(ex.source, tok, ex.lang_src, dcfg.max_chunk_tokens,
                             dcfg.max_chunks, dcfg.max_source_tokens)
        if doc is None:
            stats["drop_empty_source"] += 1
            continue
        if doc.n_dropped_tokens > 0:
            stats["drop_source_truncated"] += 1
            continue
        # the target is encoded too (contrastive positives are source<->partner
        # document vectors), so it needs chunk spans and the same caps
        tgt_doc = chunk_document(ex.target, tok, ex.lang_tgt, dcfg.max_chunk_tokens,
                                 dcfg.max_chunks, dcfg.max_target_tokens)
        if tgt_doc is None or tgt_doc.n_dropped_tokens > 0:
            stats["drop_target_length"] += 1
            continue
        tgt_ids = tgt_doc.input_ids
        neg_texts, neg_ids, neg_spans = [], [], []
        for ntext in ex.negatives[:2]:
            nd = chunk_document(ntext, tok, ex.lang_src, dcfg.max_chunk_tokens,
                                dcfg.max_chunks, dcfg.max_source_tokens)
            if nd is None or nd.n_dropped_tokens > 0:
                stats["neg_dropped"] += 1
                continue
            neg_texts.append(ntext)
            neg_ids.append(nd.input_ids)
            neg_spans.append([x for span in nd.spans for x in span])
        k_hist[doc.k] += 1
        langs[f"{ex.lang_src}->{ex.lang_tgt}"] += 1
        stats["src_tokens"] += len(doc.input_ids)
        stats["tgt_tokens"] += len(tgt_ids)
        stats["neg_attached"] += len(neg_texts)
        rows.append({
            "id": ex.id, "source_text": ex.source, "target_text": ex.target,
            "lang_src": ex.lang_src, "lang_tgt": ex.lang_tgt,
            "pair_type": ex.pair_type, "origin": ex.origin,
            "src_ids": doc.input_ids,
            "src_spans": [x for span in doc.spans for x in span],
            "tgt_ids": tgt_ids,
            "tgt_spans": [x for span in tgt_doc.spans for x in span],
            "k": doc.k, "tgt_k": tgt_doc.k,
            "n_src_tokens": len(doc.input_ids), "n_tgt_tokens": len(tgt_ids),
            "neg_texts": neg_texts, "neg_ids": neg_ids, "neg_spans": neg_spans,
        })
    return rows


def build_source(source: str, cfg, tok) -> dict:
    from datasets import Dataset

    out_dir = OUT / source
    if out_dir.exists():
        print(f"[skip] {source}: shard exists")
        return {}
    pcfg = cfg.data.pilot
    stats, k_hist, langs = Counter(), Counter(), Counter()
    t0 = time.monotonic()
    examples = list(extract(source, pcfg, stats))
    stats["extracted"] = len(examples)
    add_generated_negatives(examples, pcfg, stats)
    rows = pack(examples, tok, cfg.data, stats, k_hist, langs)
    stats["packed"] = len(rows)
    Dataset.from_list(rows, features=features()).save_to_disk(str(out_dir))
    info = {
        "stats": dict(stats), "k_hist": dict(sorted(k_hist.items())),
        "langs": dict(langs.most_common()), "seconds": int(time.monotonic() - t0),
    }
    (out_dir / "build_stats.json").write_text(json.dumps(info, indent=2))
    k3 = sum(v for k, v in k_hist.items() if k >= 3)
    print(f"[done] {source}: {len(rows):,} pairs, {stats['src_tokens'] + stats['tgt_tokens']:,} tokens, "
          f"K>=3 {k3 / max(len(rows), 1):.1%}, {info['seconds']}s")
    return info


def finalize(cfg) -> None:
    from datasets import concatenate_datasets, load_from_disk

    shards, agg, k_all, langs_all = [], Counter(), Counter(), Counter()
    per_source = {}
    for d in sorted(OUT.iterdir()):
        if not (d / "build_stats.json").exists():
            continue
        info = json.loads((d / "build_stats.json").read_text())
        per_source[d.name] = info
        shards.append(load_from_disk(str(d)))
        agg.update({k: v for k, v in info["stats"].items() if isinstance(v, int)})
        k_all.update({int(k): v for k, v in info["k_hist"].items()})
        langs_all.update(info["langs"])
    full = concatenate_datasets(shards).shuffle(seed=cfg.data.pilot["seed"])
    full.save_to_disk(str(OUT / "full"))
    n = len(full)
    k3 = sum(v for k, v in k_all.items() if k >= 3)
    summary = {
        "n_pairs": n,
        "total_tokens_src_tgt": agg["src_tokens"] + agg["tgt_tokens"],
        "multichunk_share_pairs": round(k3 / n, 4),
        "target_multichunk_share": cfg.data.pilot["target_multichunk_share"],
        "k_hist": dict(sorted(k_all.items())),
        "langs": dict(langs_all.most_common()),
        "aggregate_stats": dict(agg),
        "per_source": per_source,
    }
    Path("runs/m1").mkdir(parents=True, exist_ok=True)
    Path("runs/m1/pilot_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_source", "k_hist", "langs")}, indent=2))
    ok = summary["multichunk_share_pairs"] >= cfg.data.pilot["target_multichunk_share"]
    print(f"K>=3 share {summary['multichunk_share_pairs']:.1%} vs target "
          f"{cfg.data.pilot['target_multichunk_share']:.0%} -> {'OK' if ok else 'BELOW TARGET'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--source", default=None, help="build one source shard")
    ap.add_argument("--finalize", action="store_true", help="only concatenate existing shards")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.data.pilot is None:
        raise SystemExit("configs: data.pilot section missing")
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.finalize:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
        todo = [args.source] if args.source else list(cfg.data.pilot["sources"])
        for source in todo:
            build_source(source, cfg, tok)
    if args.finalize or not args.source:
        finalize(cfg)


if __name__ == "__main__":
    main()
