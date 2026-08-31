"""Collator unit tests (PHASE1_PLAN §7 M1). Rows built with the real tokenizer
and chunker — no network, tokenizer comes from the local HF cache."""

import pytest
import torch
from transformers import AutoTokenizer

from abstractnet.data.chunking import chunk_document
from abstractnet.data.collate import LANG_IDX, PairCollator

TEXTS = [
    ("The cat sat on the mat.", "A cat was sitting on the mat.", "en", "en"),
    (
        "The committee approved the budget. Two members voted against it. A revision follows next month.",
        "Le comité a approuvé le budget. Deux membres ont voté contre. Une révision suivra le mois prochain.",
        "en", "fr",
    ),
    (
        "Alice met Bob. They discussed the merger. The talks lasted three hours. Nothing was signed. Both left early.",
        "Alice and Bob met to discuss the merger; after three hours of talks nothing was signed and both left early.",
        "en", "en",
    ),
]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")


@pytest.fixture(scope="module")
def rows(tok):
    out = []
    for i, (src, tgt, ls, lt) in enumerate(TEXTS):
        sd = chunk_document(src, tok, ls)
        td = chunk_document(tgt, tok, lt)
        negs = []
        if i == 2:  # one row carries a negative
            nd = chunk_document(src.replace("three hours", "five hours"), tok, ls)
            negs.append(nd)
        out.append({
            "id": f"r{i}", "pair_type": "paraphrase", "origin": "test",
            "lang_src": ls, "lang_tgt": lt,
            "src_ids": sd.input_ids, "src_spans": [x for s in sd.spans for x in s],
            "tgt_ids": td.input_ids, "tgt_spans": [x for s in td.spans for x in s],
            "neg_ids": [n.input_ids for n in negs],
            "neg_spans": [[x for s in n.spans for x in s] for n in negs],
        })
    return out


def pad_id(tok):
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def test_shapes_and_masks(tok, rows):
    b = PairCollator(pad_id(tok))(rows)
    B, S = b["src_ids"].shape
    assert B == 3
    K = b["src_chunk_mask"].shape[1]
    assert b["src_chunk_mask"].shape == (B, K, S)
    assert b["src_z_mask"].shape == (B, K)
    # chunks partition exactly the real (unpadded) tokens
    assert torch.equal(b["src_chunk_mask"].any(dim=1), b["src_mask"])
    # chunks are disjoint
    assert (b["src_chunk_mask"].int().sum(dim=1) <= 1).all()
    # z_mask counts = chunk counts
    for i, r in enumerate(rows):
        assert b["src_z_mask"][i].sum().item() == len(r["src_spans"]) // 2
    assert torch.equal(b["tgt_chunk_mask"].any(dim=1), b["tgt_mask"])


def test_labels_ignore_padding(tok, rows):
    b = PairCollator(pad_id(tok))(rows)
    assert (b["labels"][~b["tgt_mask"]] == -100).all()
    assert (b["labels"][b["tgt_mask"]] != -100).all()
    assert torch.equal(b["labels"][b["tgt_mask"]], b["tgt_ids"][b["tgt_mask"]])


def test_negatives_flattened_with_owner(tok, rows):
    b = PairCollator(pad_id(tok))(rows)
    assert b["neg_ids"].shape[0] == 1
    assert b["neg_owner"].tolist() == [2]
    assert torch.equal(b["neg_chunk_mask"].any(dim=1), b["neg_mask"])


def test_no_negatives_edge(tok, rows):
    slim = [dict(r, neg_ids=[], neg_spans=[]) for r in rows]
    b = PairCollator(pad_id(tok))(slim)
    assert b["neg_ids"].shape[0] == 0 and b["neg_owner"].numel() == 0


def test_lang_indices(tok, rows):
    b = PairCollator(pad_id(tok))(rows)
    assert b["lang_tgt_idx"].tolist() == [LANG_IDX["en"], LANG_IDX["fr"], LANG_IDX["en"]]


def test_identity_swap(tok, rows):
    b = PairCollator(pad_id(tok), p_id=1.0)(rows)
    assert b["is_identity"].all()
    # targets replaced by sources: same ids where unpadded, source language used
    for i, r in enumerate(rows):
        n = len(r["src_ids"])
        assert b["tgt_ids"][i, :n].tolist() == r["src_ids"]
        assert b["lang_tgt_idx"][i].item() == LANG_IDX[r["lang_src"]]
    b0 = PairCollator(pad_id(tok), p_id=0.0)(rows)
    assert not b0["is_identity"].any()
    assert b0["tgt_ids"][1, :3].tolist() == rows[1]["tgt_ids"][:3]


def test_deterministic_without_identity(tok, rows):
    a = PairCollator(pad_id(tok))(rows)
    b = PairCollator(pad_id(tok))(rows)
    for key in ("src_ids", "tgt_ids", "labels", "src_chunk_mask"):
        assert torch.equal(a[key], b[key])
