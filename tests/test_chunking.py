"""Chunking unit tests (PHASE1_PLAN §7 M1). Uses the real Qwen3 tokenizer from cache."""

import pytest
from transformers import AutoTokenizer

from abstractnet.data.chunking import chunk_document

EN3 = (
    "The committee approved the budget on Tuesday. "
    "However, two members voted against the proposal because it lacked detail. "
    "A revised version will be submitted next month."
)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")


def assert_partition(doc, max_chunk_tokens=64):
    assert doc.spans[0][0] == 0
    assert doc.spans[-1][1] == len(doc.input_ids)
    for (s, e), (s2, _) in zip(doc.spans, doc.spans[1:]):
        assert e == s2, "spans must be contiguous"
    for s, e in doc.spans:
        assert 1 <= e - s <= max_chunk_tokens


def test_english_three_sentences(tok):
    doc = chunk_document(EN3, tok, "en")
    assert doc.k == 3
    assert_partition(doc)
    assert doc.n_dropped_tokens == 0


def test_round_trip_text(tok):
    doc = chunk_document(EN3, tok, "en")
    assert tok.decode(doc.input_ids) == EN3
    # each chunk decodes to a piece of the original, in order
    joined = "".join(tok.decode(doc.input_ids[s:e]) for s, e in doc.spans)
    assert joined == EN3


def test_long_sentence_is_hard_split(tok):
    text = "The value is " + " and ".join(f"x{i}" for i in range(200)) + "."
    doc = chunk_document(text, tok, "en", max_chunk_tokens=64, max_chunks=8, max_tokens=512)
    assert doc.k > 1
    assert_partition(doc)


def test_caps_max_chunks_and_tokens(tok):
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(40))
    doc = chunk_document(text, tok, "en")
    assert doc.k <= 8
    assert len(doc.input_ids) <= 512
    assert doc.n_dropped_tokens > 0
    assert_partition(doc)


def test_french_and_german(tok):
    fr = "Le comité a approuvé le budget mardi. Deux membres ont voté contre. Une version révisée suivra."
    de = "Der Ausschuss billigte den Haushalt am Dienstag. Zwei Mitglieder stimmten dagegen. Eine neue Fassung folgt."
    for text, lang in ((fr, "fr"), (de, "de")):
        doc = chunk_document(text, tok, lang)
        assert doc.k == 3, f"{lang}: expected 3 chunks, got {doc.k}"
        assert_partition(doc)


def test_fallback_language(tok):
    # 'pt' has no pysbd rules -> punctuation fallback must still chunk sanely
    pt = "O comité aprovou o orçamento na terça-feira. Dois membros votaram contra. Uma versão revista será apresentada."
    doc = chunk_document(pt, tok, "pt")
    assert doc.k == 3
    assert_partition(doc)


def test_empty_and_whitespace(tok):
    assert chunk_document("", tok, "en") is None
    assert chunk_document("   \n  ", tok, "en") is None


def test_deterministic(tok):
    a = chunk_document(EN3, tok, "en")
    b = chunk_document(EN3, tok, "en")
    assert a.input_ids == b.input_ids and a.spans == b.spans
