"""Batch collation for AbstractLM training (PHASE1_PLAN §2.1 output contract).

Consumes rows from the pilot dataset (src/tgt ids + flattened chunk spans +
negatives) and produces padded tensors:

  src_ids [B,S] int64        src_mask [B,S] bool
  src_chunk_mask [B,K,S]     src_z_mask [B,K]        (K = max chunks in batch)
  tgt_ids [B,T]              tgt_mask [B,T]
  tgt_chunk_mask [B,Kt,T]    tgt_z_mask [B,Kt]       (target is encoded too:
                                                      contrastive positives)
  labels [B,T]               (-100 at padding; the model shifts for CE)
  lang_tgt_idx [B]           is_identity [B] bool
  neg_ids [N,Sn] neg_mask neg_chunk_mask neg_z_mask  (all negatives, flattened)
  neg_owner [N]              (batch index each negative belongs to)
  meta: ids / pair_type / origin lists (non-tensor)

Identity targets (p_id, PHASE1_PLAN §3): with probability p_id the target slots
are replaced by the SOURCE (ids, spans, language) and is_identity is set — the
train loop schedules p_id by setting `collator.p_id` (we run num_workers=0, so
the attribute is live; recorded in CLAUDE.md). Word dropout is model-side
(train-only, after the lang token is prepended), not collation.
"""

from __future__ import annotations

import random

import torch

LANGS = ["en", "fr", "de", "es", "it", "pt"]
LANG_IDX = {l: i for i, l in enumerate(LANGS)}


def _unflatten(spans: list[int]) -> list[tuple[int, int]]:
    return [(spans[i], spans[i + 1]) for i in range(0, len(spans), 2)]


def _pad_docs(docs: list[tuple[list[int], list[int]]], pad_id: int):
    """docs: (ids, flattened spans) per item -> padded ids/mask/chunk_mask/z_mask."""
    n = len(docs)
    S = max((len(ids) for ids, _ in docs), default=1)
    K = max((len(sp) // 2 for _, sp in docs), default=1)
    ids_t = torch.full((n, S), pad_id, dtype=torch.long)
    mask = torch.zeros(n, S, dtype=torch.bool)
    chunk_mask = torch.zeros(n, K, S, dtype=torch.bool)
    z_mask = torch.zeros(n, K, dtype=torch.bool)
    for i, (ids, flat) in enumerate(docs):
        ids_t[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, : len(ids)] = True
        for k, (s, e) in enumerate(_unflatten(flat)):
            chunk_mask[i, k, s:e] = True
            z_mask[i, k] = True
    return ids_t, mask, chunk_mask, z_mask


class PairCollator:
    def __init__(self, pad_id: int, p_id: float = 0.0, seed: int = 0):
        self.pad_id = pad_id
        self.p_id = p_id  # scheduled from the train loop
        self._rng = random.Random(seed)

    def reseed(self, seed: int) -> None:
        """Make the identity draws a pure function of the caller's seed — the
        train loop reseeds per (run_seed, step, micro) so a resumed run makes
        bit-identical identity decisions (M3 resume proof)."""
        self._rng = random.Random(seed)

    def __call__(self, rows: list[dict]) -> dict:
        srcs = [(r["src_ids"], r["src_spans"]) for r in rows]
        tgts, langs, is_identity = [], [], []
        for r in rows:
            if self._rng.random() < self.p_id:
                tgts.append((r["src_ids"], r["src_spans"]))
                langs.append(r["lang_src"])
                is_identity.append(True)
            else:
                tgts.append((r["tgt_ids"], r["tgt_spans"]))
                langs.append(r["lang_tgt"])
                is_identity.append(False)

        src_ids, src_mask, src_chunk_mask, src_z_mask = _pad_docs(srcs, self.pad_id)
        tgt_ids, tgt_mask, tgt_chunk_mask, tgt_z_mask = _pad_docs(tgts, self.pad_id)
        labels = tgt_ids.masked_fill(~tgt_mask, -100)

        negs, owner = [], []
        for i, r in enumerate(rows):
            for ids, flat in zip(r["neg_ids"], r["neg_spans"]):
                negs.append((ids, flat))
                owner.append(i)
        if negs:
            neg_ids, neg_mask, neg_chunk_mask, neg_z_mask = _pad_docs(negs, self.pad_id)
        else:
            neg_ids = torch.zeros(0, 1, dtype=torch.long)
            neg_mask = torch.zeros(0, 1, dtype=torch.bool)
            neg_chunk_mask = torch.zeros(0, 1, 1, dtype=torch.bool)
            neg_z_mask = torch.zeros(0, 1, dtype=torch.bool)

        return {
            "src_ids": src_ids, "src_mask": src_mask,
            "src_chunk_mask": src_chunk_mask, "src_z_mask": src_z_mask,
            "tgt_ids": tgt_ids, "tgt_mask": tgt_mask,
            "tgt_chunk_mask": tgt_chunk_mask, "tgt_z_mask": tgt_z_mask,
            "labels": labels,
            "lang_tgt_idx": torch.tensor([LANG_IDX[l] for l in langs], dtype=torch.long),
            "is_identity": torch.tensor(is_identity, dtype=torch.bool),
            "neg_ids": neg_ids, "neg_mask": neg_mask,
            "neg_chunk_mask": neg_chunk_mask, "neg_z_mask": neg_z_mask,
            "neg_owner": torch.tensor(owner, dtype=torch.long),
            "meta": {
                "id": [r["id"] for r in rows],
                "pair_type": [r["pair_type"] for r in rows],
                "origin": [r["origin"] for r in rows],
            },
        }
