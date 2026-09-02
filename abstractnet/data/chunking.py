"""Sentence chunking → token spans (PHASE1_PLAN §2.1, encoder step 3).

v1 chunks are sentences (pysbd where the language is supported, a punctuation
fallback otherwise); any sentence longer than max_chunk_tokens is hard-split;
documents are capped at max_chunks chunks / max_tokens tokens. Spans are
computed against the tokenizer's offset_mapping, so they line up with the exact
input_ids the encoder sees. Learned segmentation is a later ablation, not v1.

Span contract (consumed by the pooling head and the collator):
  spans partition [0, len(input_ids)) — contiguous, ordered, non-overlapping,
  first starts at 0, last ends at len(input_ids), each of length
  1..max_chunk_tokens.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

# Languages our pilot mix uses that pysbd actually ships rules for. Everything
# else takes the punctuation fallback (recorded per-source in DATA.md).
_PYSBD_LANGS = {"en", "fr", "de", "es", "it", "nl", "da", "el", "ru", "pl", "bg", "ar", "zh", "ja"}

_FALLBACK_SENT_RE = re.compile(r"[^.!?\n]*[.!?\n]+[\s]*|[^.!?\n]+$")


@dataclass
class ChunkedDoc:
    input_ids: list[int]
    spans: list[tuple[int, int]]  # [start, end) token indices
    n_dropped_tokens: int  # tokens beyond the caps that were truncated away

    @property
    def k(self) -> int:
        return len(self.spans)


@lru_cache(maxsize=None)
def _segmenter(lang: str):
    if lang in _PYSBD_LANGS:
        import pysbd

        try:
            return pysbd.Segmenter(language=lang, clean=False, char_span=True)
        except Exception:
            return None
    return None


def sentence_starts(text: str, lang: str) -> list[int]:
    """Character offsets where sentences start; always begins with 0.
    Sentence i owns chars [starts[i], starts[i+1])."""
    seg = _segmenter(lang)
    if seg is not None:
        spans = seg.segment(text)
        starts = [s.start for s in spans if s.sent.strip()]
    else:
        starts = [m.start() for m in _FALLBACK_SENT_RE.finditer(text) if m.group().strip()]
    if not starts or starts[0] != 0:
        starts = [0] + [s for s in starts if s != 0]
    return starts


def visible_sentence_count(
    text: str,
    tokenizer,
    lang: str,
    max_chunk_tokens: int = 64,
    max_chunks: int = 8,
    max_tokens: int = 512,
) -> int:
    """How many leading sentences survive the caps IN FULL (v1.3 guard: a
    perturbation beyond this index is model-invisible — the impossible-negative
    bug found at pilot step 500)."""
    doc = chunk_document(text, tokenizer, lang, max_chunk_tokens, max_chunks, max_tokens)
    if doc is None:
        return 0
    if doc.n_dropped_tokens == 0:
        return len(sentence_starts(text, lang))
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    cut_char = enc["offset_mapping"][len(doc.input_ids) - 1][1]
    starts = sentence_starts(text, lang)
    ends = starts[1:] + [len(text)]
    return sum(1 for e in ends if e <= cut_char)


def chunk_document(
    text: str,
    tokenizer,
    lang: str,
    max_chunk_tokens: int = 64,
    max_chunks: int = 8,
    max_tokens: int = 512,
) -> ChunkedDoc | None:
    """Chunk one document. Returns None for empty/whitespace text."""
    if not text or not text.strip():
        return None
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if not ids:
        return None

    starts = sentence_starts(text, lang)

    # sentence id per token, by the char offset where the token starts
    sent_of_token, si = [], 0
    for cs, _ in offsets:
        while si + 1 < len(starts) and cs >= starts[si + 1]:
            si += 1
        sent_of_token.append(si)

    # contiguous token ranges per sentence, long sentences hard-split
    ranges: list[tuple[int, int]] = []
    range_start = 0
    for t in range(1, len(ids) + 1):
        if t == len(ids) or sent_of_token[t] != sent_of_token[range_start]:
            for s in range(range_start, t, max_chunk_tokens):
                ranges.append((s, min(s + max_chunk_tokens, t)))
            range_start = t

    # apply the document caps
    spans: list[tuple[int, int]] = []
    for s, e in ranges:
        if len(spans) == max_chunks or e > max_tokens:
            break
        spans.append((s, e))
    if not spans:  # first sentence alone exceeded max_tokens
        spans = [(0, min(max_chunk_tokens, len(ids), max_tokens))]

    cut = spans[-1][1]
    return ChunkedDoc(input_ids=ids[:cut], spans=spans, n_dropped_tokens=len(ids) - cut)
