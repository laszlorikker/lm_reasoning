"""M3.1 packing tests: budgets, rotation, determinism, exact state resume,
and the contrastive negative band. Synthetic dataset — no network, no GPU."""

import numpy as np
import pytest
import torch
from datasets import Dataset

from abstractnet.data.packing import BucketedSampler
from abstractnet.modeling.losses import symmetric_info_nce


def make_ds(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    ts = rng.integers(15, 500, n)
    tt = rng.integers(15, 500, n)
    return Dataset.from_dict({
        "id": [f"r{i:04d}" for i in range(n)],
        "n_src_tokens": ts.tolist(), "n_tgt_tokens": tt.tolist(),
        "src_ids": [[1] * int(t) for t in ts],
        "tgt_ids": [[1] * int(t) for t in tt],
    })


def sampler(ds, **kw):
    args = dict(seed=11, n_buckets=4, micro_token_budget=2500,
                max_rows_per_micro=16, token_budget=8000)
    args.update(kw)
    return BucketedSampler(ds, **args)


def test_plan_respects_budgets():
    s = sampler(make_ds())
    plan = s.plan_step()
    total = 0
    for micro in plan:
        assert 1 <= len(micro) <= 16
        toks = sum(len(r["src_ids"]) + len(r["tgt_ids"]) for r in micro)
        assert toks <= 2500 or len(micro) == 1  # single oversized row allowed
        total += toks
    assert total >= 8000


def test_micros_are_single_bucket_and_rotate():
    ds = make_ds()
    s = sampler(ds)
    plan = s.plan_step()
    buckets_seen = []
    for micro in plan:
        ids = [int(r["id"][1:]) for r in micro]
        bs = {int(s.bucket_of[i]) for i in ids}
        assert len(bs) == 1, "each micro must come from ONE bucket"
        buckets_seen.append(bs.pop())
    assert buckets_seen == [i % 4 for i in range(len(buckets_seen))], \
        "buckets must rotate across the accumulation cycle"
    assert len(set(buckets_seen)) >= 2, "no optimizer step may be single-bucket"


def test_deterministic():
    a = sampler(make_ds()).plan_step()
    b = sampler(make_ds()).plan_step()
    assert [[r["id"] for r in m] for m in a] == [[r["id"] for r in m] for m in b]


def test_state_round_trip_resume_exact():
    ds = make_ds()
    s1 = sampler(ds)
    for _ in range(3):
        s1.plan_step()
    snap = s1.state()
    ref = [[[r["id"] for r in m] for m in s1.plan_step()] for _ in range(2)]
    s2 = sampler(ds)
    s2.load_state(snap)
    got = [[[r["id"] for r in m] for m in s2.plan_step()] for _ in range(2)]
    assert ref == got


def test_epoch_wraps_without_repeat_within_epoch():
    ds = make_ds(n=120)
    s = sampler(ds, token_budget=4000)
    seen: dict[int, list[str]] = {}
    for _ in range(30):  # forces per-bucket epoch wraps
        for micro in s.plan_step():
            for r in micro:
                b = int(s.bucket_of[int(r["id"][1:])])
                seen.setdefault(b, []).append(r["id"])
    for b, ids in seen.items():
        n_rows = len(s.rows_by_bucket[b])
        for e in range(len(ids) // n_rows):
            chunk = ids[e * n_rows:(e + 1) * n_rows]
            assert len(set(chunk)) == n_rows, f"bucket {b} epoch {e} repeats a row"


def test_negative_band_cap():
    torch.manual_seed(0)
    va = torch.nn.functional.normalize(torch.randn(40, 16), dim=-1)
    vb = torch.nn.functional.normalize(torch.randn(40, 16), dim=-1)
    vn = torch.nn.functional.normalize(torch.randn(5, 16), dim=-1)
    full = symmetric_info_nce(va, vb, vn, 0.05)
    capped1 = symmetric_info_nce(va, vb, vn, 0.05, max_inbatch_negs=31, seed=7)
    capped2 = symmetric_info_nce(va, vb, vn, 0.05, max_inbatch_negs=31, seed=7)
    other = symmetric_info_nce(va, vb, vn, 0.05, max_inbatch_negs=31, seed=8)
    assert torch.isfinite(capped1)
    assert torch.equal(capped1, capped2), "cap must be deterministic in seed"
    assert not torch.equal(capped1, other), "different seed, different subsample"
    assert not torch.equal(full, capped1), "cap must actually bind at B=40"
    # below the band, the cap is a no-op
    small = symmetric_info_nce(va[:8], vb[:8], vn, 0.05)
    small_c = symmetric_info_nce(va[:8], vb[:8], vn, 0.05, max_inbatch_negs=31, seed=7)
    assert torch.equal(small, small_c)
