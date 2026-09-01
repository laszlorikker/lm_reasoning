"""Pair construction for the pilot corpus (PHASE1_PLAN §3 + M1 additions 2/4).

Each training example is (source, target, lang_tgt, negatives[]). Positives:
paraphrase / translation / NLI-entailment / same-problem math derivations.
Multi-chunk positives (target ≥30% of pairs with K≥3) come from three places:
consecutive aligned rows of document-ordered parallel corpora (europarl,
news_commentary — row order verified continuous), synthetic concatenation of
independent paraphrase pairs (same order both sides, so the pair is meaning-
equivalent and chunk-aligned), and naturally multi-sentence math solutions.

The identity-target probability p_id is applied at TRAIN time (M3), not here.

Val separation: paws-x/qqp/mrpc/mnli/opus-100 have held-out splits; concat
sources reserve the first VAL_RESERVED_DOCS documents per config; math reserves
problems by stable hash (is_val_problem).
"""

from __future__ import annotations

import hashlib
import itertools
import random
from collections import OrderedDict
from dataclasses import dataclass, field

from abstractnet.data.minimal_pairs import hash_seed

OPUS_CFGS = ["de-en", "en-es", "en-fr", "en-it", "en-pt"]
EUROPARL_CFGS = ["en-fr", "de-en", "en-es", "en-it", "en-pt"]
NEWSCOMM_CFGS = ["en-fr", "de-en", "en-es"]
VAL_RESERVED_DOCS = 3000  # first N concat docs per config are val-only
_TERMINAL = tuple(".!?…\"')")


@dataclass
class PairExample:
    source: str
    target: str
    lang_src: str
    lang_tgt: str
    pair_type: str  # translation | paraphrase | nli_entailment | math_derivation | concat_translation | concat_paraphrase
    origin: str
    negatives: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return hashlib.sha1(f"{self.origin}|{self.source}|{self.target}".encode()).hexdigest()[:16]


def _wc(s: str) -> int:
    return len(s.split())


def _ok_sent(s: str, lo: int = 3, hi: int = 80) -> bool:
    return lo <= _wc(s) <= hi


def is_val_problem(problem: str) -> bool:
    return hash_seed(problem) % 97 == 0


# ---------------------------------------------------------------- translation


def iter_opus100_singles(n: int, seed: int):
    import datasets

    rng = random.Random(seed)
    per = n // len(OPUS_CFGS)
    for cfg in OPUS_CFGS:
        a, b = cfg.split("-")
        ds = datasets.load_dataset("Helsinki-NLP/opus-100", cfg, split="train", streaming=True)
        got = 0
        for row in ds:
            sa, sb = row["translation"][a].strip(), row["translation"][b].strip()
            if not (_ok_sent(sa) and _ok_sent(sb)):
                continue
            if not (1 / 3 < max(_wc(sa), 1) / max(_wc(sb), 1) < 3):
                continue
            if rng.random() < 0.5:
                src, tgt, ls, lt = sa, sb, a, b
            else:
                src, tgt, ls, lt = sb, sa, b, a
            yield PairExample(src, tgt, ls, lt, "translation", f"opus100-{cfg}")
            got += 1
            if got >= per:
                break


def _concat_rows(rows: list[dict], a: str, b: str, stats) -> tuple[str, str] | None:
    """Join k consecutive aligned rows into one aligned document pair."""
    pa, pb = [], []
    for r in rows:
        sa, sb = r["translation"][a].strip(), r["translation"][b].strip()
        if not (_ok_sent(sa, 2, 70) and _ok_sent(sb, 2, 70)):
            return None
        if not sa.endswith(_TERMINAL):
            sa += "."
            stats["punct_fixed"] += 1
        if not sb.endswith(_TERMINAL):
            sb += "."
            stats["punct_fixed"] += 1
        pa.append(sa)
        pb.append(sb)
    da, db = " ".join(pa), " ".join(pb)
    if _wc(da) > 290 or _wc(db) > 290:
        return None
    return da, db


def iter_concat_translation(n: int, seed: int, k_min: int, k_max: int, stats, val: bool = False):
    import datasets

    rng = random.Random(seed)
    sources = [("Helsinki-NLP/europarl", c) for c in EUROPARL_CFGS] + [
        ("Helsinki-NLP/news_commentary", c) for c in NEWSCOMM_CFGS
    ]
    per = n // len(sources)
    for hub, cfg in sources:
        a, b = cfg.split("-")
        ds = datasets.load_dataset(hub, cfg, split="train", streaming=True)
        it = iter(ds)
        made = got = 0
        while True:
            k = rng.randint(k_min, k_max)
            rows = list(itertools.islice(it, k))
            if len(rows) < k:
                break
            doc = _concat_rows(rows, a, b, stats)
            if doc is None:
                stats["concat_skipped"] += 1
                continue
            made += 1
            in_val = made <= VAL_RESERVED_DOCS
            if in_val != val:
                continue
            da, db = doc
            if rng.random() < 0.5:
                src, tgt, ls, lt = da, db, a, b
            else:
                src, tgt, ls, lt = db, da, b, a
            yield PairExample(src, tgt, ls, lt, "concat_translation", f"{hub.split('/')[-1]}-{cfg}")
            got += 1
            if got >= per:
                break


# ---------------------------------------------------------------- paraphrase


def load_paws_pools(langs: list[str], split: str = "train") -> dict[str, list[tuple[str, str, str | None]]]:
    """Per language: (sentence1, positive_paraphrase, adversarial_negative|None).
    PAWS label 0 pairs are the adversarial ones (high overlap, different meaning)."""
    import datasets

    pools: dict[str, list] = {}
    for lang in langs:
        ds = datasets.load_dataset("google-research-datasets/paws-x", lang, split=split)
        pos: dict[str, str] = {}
        neg: dict[str, str] = {}
        for r in ds:
            s1, s2 = r["sentence1"].strip(), r["sentence2"].strip()
            if not s1 or not s2:
                continue
            (pos if r["label"] == 1 else neg).setdefault(s1, s2)
        pools[lang] = [(s1, s2, neg.get(s1)) for s1, s2 in pos.items()]
    return pools


def load_qqp_pool(split: str = "train") -> list[tuple[str, str, None]]:
    import datasets

    ds = datasets.load_dataset("nyu-mll/glue", "qqp", split=split)
    return [
        (r["question1"].strip(), r["question2"].strip(), None)
        for r in ds
        if r["label"] == 1 and r["question1"].strip() and r["question2"].strip()
    ]


def load_mrpc_pool(split: str = "train") -> list[tuple[str, str, None]]:
    import datasets

    ds = datasets.load_dataset("nyu-mll/glue", "mrpc", split=split)
    return [
        (r["sentence1"].strip(), r["sentence2"].strip(), None)
        for r in ds
        if r["label"] == 1 and r["sentence1"].strip() and r["sentence2"].strip()
    ]


def iter_paraphrase_singles(pool: list, lang: str, origin: str, n: int, seed: int):
    rng = random.Random(seed)
    for s1, s2, adv in pool[:n]:
        if rng.random() < 0.5:
            s1, s2 = s2, s1
        yield PairExample(s1, s2, lang, lang, "paraphrase", origin, negatives=[adv] if adv else [])


def iter_concat_paraphrase(pools_rest: dict[str, list], lang_shares: dict[str, float],
                           n: int, seed: int, k_min: int, k_max: int):
    """Synthetic multi-chunk paraphrase docs: concat k independent pairs, same
    order both sides. Negative (M1 addition 4): substitute ONE constituent with
    its adversarial partner when one exists."""
    rng = random.Random(seed)
    for lang, share in lang_shares.items():
        pool = list(pools_rest.get(lang, []))
        rng.shuffle(pool)
        idx, want = 0, int(n * share)
        made = 0
        while made < want and idx + k_max <= len(pool):
            k = rng.randint(k_min, k_max)
            group = pool[idx: idx + k]
            idx += k
            srcs = [g[0] for g in group]
            tgts = [g[1] for g in group]
            if any(not _ok_sent(s, 3, 70) for s in srcs + tgts):
                continue
            negatives = []
            adv_positions = [j for j, g in enumerate(group) if g[2]]
            if adv_positions:
                j = rng.choice(adv_positions)
                negatives.append(" ".join(srcs[:j] + [group[j][2]] + srcs[j + 1:]))
            yield PairExample(" ".join(srcs), " ".join(tgts), lang, lang,
                              "concat_paraphrase", f"concat-{lang}", negatives=negatives)
            made += 1


# ---------------------------------------------------------------- NLI


def iter_nli_entailment(n: int, seed: int, split: str = "train"):
    """MNLI premises that have BOTH an entailment (weak positive target) and a
    contradiction (hard negative) hypothesis. English only in v1."""
    import datasets

    ds = datasets.load_dataset("nyu-mll/multi_nli", split=split)
    by_premise: dict[str, dict[int, str]] = {}
    for r in ds:
        h = by_premise.setdefault(r["premise"].strip(), {})
        h.setdefault(r["label"], r["hypothesis"].strip())
    rng = random.Random(seed)
    items = [(p, h) for p, h in by_premise.items() if 0 in h and 2 in h and _ok_sent(p, 4, 90)]
    rng.shuffle(items)
    for premise, h in items[:n]:
        yield PairExample(premise, h[0], "en", "en", "nli_entailment", "multi_nli", negatives=[h[2]])


# ---------------------------------------------------------------- math


def iter_math_derivations_grouped(n: int, seed: int, val: bool = False,
                                  min_words: int = 15, max_words: int = 260,
                                  split: str = "train_2M", max_pairs_per_problem: int = 4):
    """M1.1 replacement for the bounded-buffer heuristic: exact group-by-problem
    join. Loads the split once (cached parquet, RAM allows it), groups guard-
    passing solutions per problem, emits up to max_pairs_per_problem disjoint
    solution pairs per problem in stable-hash problem order."""
    import datasets

    ds = datasets.load_dataset("nvidia/OpenMathInstruct-2", split=split)
    ds = ds.select_columns(["problem", "generated_solution"])
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    cap = 2 * max_pairs_per_problem
    for row in ds:
        prob = row["problem"].strip()
        if is_val_problem(prob) != val:
            continue
        sol = row["generated_solution"].strip()
        if not (min_words <= _wc(sol) <= max_words):
            continue
        g = groups.setdefault(prob, [])
        if len(g) < cap and sol not in g:
            g.append(sol)
    got = 0
    for prob in sorted(groups, key=hash_seed):
        sols = groups[prob]
        if len(sols) < 2:
            continue
        rng.shuffle(sols)
        for i in range(0, len(sols) - 1, 2):
            if got >= n:
                return
            a, b = sols[i], sols[i + 1]
            yield PairExample(a, b, "en", "en", "math_derivation", "openmath2-grouped")
            got += 1


def iter_math_derivations(n: int, seed: int, val: bool = False,
                          min_words: int = 15, max_words: int = 260,
                          split: str = "train_1M"):
    """Two solutions to the same OpenMathInstruct-2 problem = two derivations of
    one conclusion. Streaming with a bounded buffer; no full download. Val
    separation is by problem hash (is_val_problem), so it holds across splits."""
    import datasets

    ds = datasets.load_dataset("nvidia/OpenMathInstruct-2", split=split, streaming=True)
    rng = random.Random(seed)
    buffer: OrderedDict[str, str] = OrderedDict()
    got = 0
    for row in ds:
        if got >= n:
            break
        prob, sol = row["problem"].strip(), row["generated_solution"].strip()
        if not (min_words <= _wc(sol) <= max_words):
            continue
        if is_val_problem(prob) != val:
            continue
        prev = buffer.pop(prob, None)
        if prev is None:
            buffer[prob] = sol
            while len(buffer) > 100_000:
                buffer.popitem(last=False)
            continue
        if prev == sol:
            continue
        a, b = (prev, sol) if rng.random() < 0.5 else (sol, prev)
        yield PairExample(a, b, "en", "en", "math_derivation", f"openmath2-{row['problem_source']}")
        got += 1
