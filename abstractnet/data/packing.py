"""Token-budget batching (M3.1, guards a+b).

Micros are length-bucketed (equal-row quantile buckets over src+tgt tokens) and
the bucket ROTATES across the accumulation cycle, so no optimizer step is
single-bucket (guard a). An optimizer step is a fixed TOKEN budget rather than
a pair count (guard b): the step plan collects micros — each filled from one
bucket up to micro_token_budget or max_rows — until the step budget is reached,
so LR/p_id schedules keep their per-step meaning.

Deterministic and exactly resumable: per-bucket epoch-shuffled permutations
(seeded); state = per-bucket (epoch, position) + the rotation cursor. Note the
per-bucket "epoch" counters advance at different rates (short-document buckets
hold less token mass per row), which is expected and recorded.
"""

from __future__ import annotations

import numpy as np


class BucketedSampler:
    def __init__(self, ds, seed: int, n_buckets: int = 4,
                 micro_token_budget: int = 5000, max_rows_per_micro: int = 48,
                 token_budget: int = 40000):
        self.ds, self.seed = ds, seed
        self.n_buckets = n_buckets
        self.micro_budget = micro_token_budget
        self.max_rows = max_rows_per_micro
        self.token_budget = token_budget
        tok = np.asarray(ds["n_src_tokens"]) + np.asarray(ds["n_tgt_tokens"])
        self.tok = tok
        edges = np.quantile(tok, np.linspace(0, 1, n_buckets + 1)[1:-1])
        self.bucket_of = np.searchsorted(edges, tok)
        self.rows_by_bucket = {b: np.flatnonzero(self.bucket_of == b)
                               for b in range(n_buckets)}
        self.total_tokens = int(tok.sum())
        self.epoch = {b: 0 for b in range(n_buckets)}
        self.pos = {b: 0 for b in range(n_buckets)}
        self.cursor = 0
        self._perm: dict[int, np.ndarray] = {}

    # ---------------------------------------------------------------- state

    def state(self) -> dict:
        return {"mode": "packed", "seed": self.seed,
                "epoch": dict(self.epoch), "pos": dict(self.pos),
                "cursor": self.cursor}

    def load_state(self, s: dict) -> None:
        assert s.get("mode") == "packed" and s["seed"] == self.seed, \
            "checkpoint sampler state is not a packed sampler with this seed"
        self.epoch = {int(k): v for k, v in s["epoch"].items()}
        self.pos = {int(k): v for k, v in s["pos"].items()}
        self.cursor = s["cursor"]
        self._perm = {}

    # -------------------------------------------------------------- packing

    def _perm_for(self, b: int) -> np.ndarray:
        if b not in self._perm:
            rng = np.random.default_rng(self.seed + 7919 * b + 104729 * self.epoch[b])
            self._perm[b] = rng.permutation(self.rows_by_bucket[b])
        return self._perm[b]

    def _next_micro_indices(self) -> tuple[list[int], int]:
        b = self.cursor % self.n_buckets
        self.cursor += 1
        rows: list[int] = []
        tokens = 0
        while len(rows) < self.max_rows:
            perm = self._perm_for(b)
            if self.pos[b] >= len(perm):
                self.epoch[b] += 1
                self.pos[b] = 0
                self._perm.pop(b, None)
                perm = self._perm_for(b)
            idx = int(perm[self.pos[b]])
            t = int(self.tok[idx])
            if rows and tokens + t > self.micro_budget:
                break
            rows.append(idx)
            tokens += t
            self.pos[b] += 1
        return rows, tokens

    def plan_step(self) -> list[list[dict]]:
        """The full micro plan for ONE optimizer step (>= token_budget tokens).
        Row dicts are fetched here; state advances exactly one step."""
        micros: list[list[dict]] = []
        total = 0
        while total < self.token_budget:
            idxs, tokens = self._next_micro_indices()
            micros.append([dict(self.ds[i]) for i in idxs])
            total += tokens
        return micros

    def steps_per_epoch(self) -> int:
        return max(1, self.total_tokens // self.token_budget)
