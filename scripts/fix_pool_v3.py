#!/usr/bin/env python
"""Pool v3 (pilot step-500 fix): same 2,000 docs and ids as v2; hard negatives
whose CAPPED input_ids equal their source's are regenerated inside the
model-visible window (en: constrained minimal pairs; else digit-bump / drop).
Verifies zero identical negatives remain; reports per-rule chrF of the
regenerated ones. Freezes data/fixtures/val_pool_v3.jsonl. CPU-only.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abstractnet.config import load_config
from abstractnet.data.chunking import chunk_document, visible_sentence_count
from abstractnet.data.minimal_pairs import generate_minimal_pairs, hash_seed


def main() -> None:
    cfg = load_config("configs/base.yaml")
    dcfg = cfg.data
    import sacrebleu
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model.name_or_path)

    def caps_chunk(text, lang):
        return chunk_document(text, tok, lang, dcfg.max_chunk_tokens,
                              dcfg.max_chunks, dcfg.max_source_tokens)

    src = Path("data/fixtures/val_pool_v2.jsonl")
    docs = [json.loads(l) for l in src.read_text().splitlines()]
    stats = Counter()
    regen_chrf = []
    for d in docs:
        cd = caps_chunk(d["text"], d["lang"])
        new_negs = []
        for neg in d.get("hard_negatives", []):
            nd = caps_chunk(neg["text"], d["lang"])
            if nd is not None and nd.input_ids != cd.input_ids:
                new_negs.append(neg)
                continue
            stats["bad_found"] += 1
            vis = visible_sentence_count(d["text"], tok, d["lang"], dcfg.max_chunk_tokens,
                                         dcfg.max_chunks, dcfg.max_source_tokens)
            replaced = False
            if d["lang"] == "en" and vis > 0:
                cands = generate_minimal_pairs(d["text"], n=4,
                                               seed=hash_seed(d["text"]) ^ 0x13,
                                               max_sentences=vis)
                for m in cands:
                    md = caps_chunk(m.text, d["lang"])
                    if md is not None and md.input_ids != cd.input_ids:
                        new_negs.append({"text": m.text, "kind": "minimal_pair",
                                         "rule": m.rule})
                        regen_chrf.append((m.rule,
                                           sacrebleu.sentence_chrf(m.text, [d["text"]]).score))
                        stats[f"regen_{m.rule}"] += 1
                        replaced = True
                        break
            if not replaced:
                stats["dropped"] += 1
        d["hard_negatives"] = new_negs

    # re-audit inline: zero identical must remain
    remaining = 0
    for d in docs:
        cd = caps_chunk(d["text"], d["lang"])
        for neg in d["hard_negatives"]:
            nd = caps_chunk(neg["text"], d["lang"])
            if nd is not None and nd.input_ids == cd.input_ids:
                remaining += 1
    assert remaining == 0, f"{remaining} identical negatives remain"

    out = Path("data/fixtures/val_pool_v3.jsonl")
    with open(out, "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with_neg = sum(bool(d["hard_negatives"]) for d in docs)
    print(f"pool_v3 frozen: {len(docs)} docs (ids unchanged), "
          f"{with_neg} with >=1 hard negative")
    print(f"stats: {dict(stats)}")
    for rule, score in regen_chrf:
        print(f"  regenerated {rule}: chrF {score:.1f}")
    print("POOL V3 COMPLETE")


if __name__ == "__main__":
    main()
