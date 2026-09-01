"""Small helpers to pull real collated batches from a built corpus (tests,
overfit, memory profile)."""

from datasets import load_from_disk

from abstractnet.data.collate import PairCollator


def real_batch(corpus_path: str, n: int, tokenizer, seed: int = 0,
               require_negatives: bool = False, p_id: float = 0.0,
               mixed_k: bool = False, min_src_tokens: int = 0) -> dict:
    """mixed_k=True composes the batch to mirror the measured M1 K
    distribution (>=30% rows with K>=3); min_src_tokens>0 selects long
    documents (the 8x512 worst-case memory shape)."""
    ds = load_from_disk(corpus_path).shuffle(seed=seed)
    want_multi = max(1, round(0.31 * n)) if mixed_k else 0
    rows, multi = [], 0
    for r in ds:
        if require_negatives and not r["neg_texts"]:
            continue
        if r["n_src_tokens"] < min_src_tokens:
            continue
        is_multi = r["k"] >= 3
        remaining = n - len(rows)
        if mixed_k and not is_multi and remaining <= want_multi - multi:
            continue  # only multi-chunk slots left
        rows.append(dict(r))
        multi += is_multi
        if len(rows) == n:
            break
    assert len(rows) == n, f"only {len(rows)} usable rows in {corpus_path}"
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return PairCollator(pad_id=pad, p_id=p_id)(rows)
