"""Rule-based minimal-pair generator (PHASE1_PLAN §3): negatives that differ
from the source in exactly one proposition.

English-only in v1 (spaCy en_core_web_sm for dependency parses and NER). For a
multi-sentence document exactly ONE sentence is perturbed and every other
sentence is preserved byte-exact (M1 addition 4), so the contrastive term sees
documents that differ in one proposition among several.

Rules: negation flip (remove an existing negation, else insert one), number
perturbation, subject/object swap around a transitive verb, same-label entity
swap. Each rule fires cleanly or reports non-applicability for that sentence.

LLM-generated rewrites are a later addition: register a callable
(text, rng) -> str | None in GENERATORS; the builder picks them up by name.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from functools import lru_cache


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    return spacy.load("en_core_web_sm")


@dataclass
class MinimalPair:
    text: str
    rule: str
    sentence_idx: int


_NUM_WORDS = {
    "one": "two", "two": "three", "three": "four", "four": "five", "five": "six",
    "six": "seven", "seven": "eight", "eight": "nine", "nine": "ten",
    "ten": "eleven", "eleven": "twelve", "twelve": "ten",
}


def _rule_negation(sent, rng: random.Random) -> str | None:
    """Negation flip. Insertion is grammar-aware (M1.1): under subject–aux
    inversion (questions) "not" goes AFTER the subject ("What do you not
    think…"), never between aux and subject; sentence-initial verbs
    (imperatives) are skipped — the generator falls back to other rules."""
    s = sent.text
    for t in sent:
        if t.dep_ == "neg":  # remove an existing negation
            a = t.idx - sent.start_char
            b = a + len(t.text)
            if a > 0 and s[a - 1] == " ":
                a -= 1
            elif b < len(s) and s[b] == " ":
                b += 1
            return s[:a] + s[b:]
    root = sent.root
    subj = next((c for c in root.children if c.dep_ in ("nsubj", "nsubjpass", "expl")), None)
    auxes = [c for c in root.children if c.dep_ in ("aux", "auxpass", "cop")]
    if root.pos_ == "AUX" or (root.pos_ == "VERB" and root.lemma_ == "be"):
        anchor = root
    elif auxes:
        anchor = auxes[0]
    elif root.pos_ == "VERB":
        if root.i == sent.start:  # imperative / verb-initial: no clean negation
            return None
        a = root.idx - sent.start_char
        return s[:a] + "never " + s[a:]
    else:
        return None
    if subj is not None and subj.i > anchor.i:  # inversion: not after the subject
        b = max(t.idx + len(t.text) for t in subj.subtree) - sent.start_char
        return s[:b] + " not" + s[b:]
    b = anchor.idx - sent.start_char + len(anchor.text)
    return s[:b] + " not" + s[b:]


def _rule_number(sent, rng: random.Random) -> str | None:
    s = sent.text
    nums = [t for t in sent if t.pos_ == "NUM" or t.like_num]
    rng.shuffle(nums)
    for t in nums:
        rel = t.idx - sent.start_char
        txt = t.text
        m = re.fullmatch(r"(\d+)(\.\d+)?", txt)
        if m:
            val = int(m.group(1))
            delta = rng.randint(1, 9)
            new_int = val + delta if (rng.random() < 0.5 or val - delta < 0) else val - delta
            new = str(new_int) + (m.group(2) or "")
            if new != txt:
                return s[:rel] + new + s[rel + len(txt):]
        lw = txt.lower()
        if lw in _NUM_WORDS:
            new = _NUM_WORDS[lw]
            if txt[0].isupper():
                new = new.capitalize()
            return s[:rel] + new + s[rel + len(txt):]
    return None


def _subtree_span(tok, sent) -> tuple[int, int]:
    toks = list(tok.subtree)
    a = min(t.idx for t in toks) - sent.start_char
    b = max(t.idx + len(t.text) for t in toks) - sent.start_char
    return a, b


def _rule_arg_swap(sent, rng: random.Random) -> str | None:
    s = sent.text
    verbs = [t for t in sent if t.pos_ == "VERB"]
    rng.shuffle(verbs)
    for v in verbs:
        subj = next((c for c in v.children if c.dep_ == "nsubj"), None)
        obj = next((c for c in v.children if c.dep_ in ("dobj", "obj")), None)
        if subj is None or obj is None:
            continue
        (a1, b1), (a2, b2) = _subtree_span(subj, sent), _subtree_span(obj, sent)
        if b1 >= a2:  # overlapping or reversed spans: malformed parse, skip
            continue
        t1, t2 = s[a1:b1], s[a2:b2]
        if t1 == t2:
            continue
        # move determiner capitalisation with the sentence position, not the phrase
        if t1[0].isupper() and t2[0].islower() and subj.subtree.__iter__().__next__().pos_ == "DET":
            t1 = t1[0].lower() + t1[1:]
            t2 = t2[0].upper() + t2[1:]
        return s[:a1] + t2 + s[b1:a2] + t1 + s[b2:]
    return None


def _rule_entity_swap(sent, rng: random.Random) -> str | None:
    s = sent.text
    by_label: dict[str, list] = {}
    for e in sent.ents:
        if e.label_ in ("PERSON", "ORG", "GPE", "LOC"):
            by_label.setdefault(e.label_, []).append(e)
    labels = [l for l, es in by_label.items() if len(es) >= 2 and es[0].text != es[1].text]
    if not labels:
        return None
    e1, e2 = by_label[rng.choice(sorted(labels))][:2]
    a1, b1 = e1.start_char - sent.start_char, e1.end_char - sent.start_char
    a2, b2 = e2.start_char - sent.start_char, e2.end_char - sent.start_char
    return s[:a1] + e2.text + s[b1:a2] + e1.text + s[b2:]


RULES: list[tuple[str, object]] = [
    ("negation", _rule_negation),
    ("number", _rule_number),
    ("arg_swap", _rule_arg_swap),
    ("entity_swap", _rule_entity_swap),
]

# Hook for later LLM-generated rewrites/negatives (PHASE1_PLAN §3):
# name -> callable(full_text, rng) -> str | None. Not used by v1 rules.
GENERATORS: dict[str, object] = {}


def generate_minimal_pairs(text: str, n: int = 2, seed: int = 0) -> list[MinimalPair]:
    """Up to n distinct minimal-pair negatives for one (possibly multi-sentence)
    English document. Deterministic in (text, n, seed)."""
    return _pairs_from_parsed(_nlp()(text), text, n, seed)


def generate_minimal_pairs_bulk(
    texts: list[str], n: int = 1, seeds: list[int] | None = None, batch_size: int = 128
) -> list[list[MinimalPair]]:
    """Batched variant for the data builder: one spaCy pipe pass over all texts.
    Deterministic per text (seed defaults to a stable hash of the text), so the
    result does not depend on batch composition."""
    if seeds is None:
        seeds = [hash_seed(t) for t in texts]
    docs = _nlp().pipe(texts, batch_size=batch_size)
    return [_pairs_from_parsed(d, t, n, s) for d, t, s in zip(docs, texts, seeds)]


def hash_seed(text: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha1(text.encode()).digest()[:4], "big")


def _pairs_from_parsed(doc, text: str, n: int, seed: int) -> list[MinimalPair]:
    sents = list(doc.sents)
    rng = random.Random(seed)
    combos = [(i, name, fn) for i in range(len(sents)) for name, fn in RULES]
    rng.shuffle(combos)
    out: list[MinimalPair] = []
    seen = {text}
    for i, name, fn in combos:
        if len(out) >= n:
            break
        new_sent = fn(sents[i], rng)
        if not new_sent or not new_sent.strip() or new_sent == sents[i].text:
            continue
        new_doc = text[: sents[i].start_char] + new_sent + text[sents[i].end_char:]
        if new_doc in seen:
            continue
        seen.add(new_doc)
        out.append(MinimalPair(text=new_doc, rule=name, sentence_idx=i))
    return out
