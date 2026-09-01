"""Minimal-pair generator tests (PHASE1_PLAN §7 M1, addition 4) on crafted examples."""

import random
import re

import pytest

from abstractnet.data.minimal_pairs import (
    MinimalPair,
    _nlp,
    _rule_arg_swap,
    _rule_entity_swap,
    _rule_negation,
    _rule_number,
    generate_minimal_pairs,
)


def sent_of(text):
    return list(_nlp()(text).sents)[0]


def rng():
    return random.Random(0)


def test_negation_removal():
    out = _rule_negation(sent_of("The committee did not approve the budget."), rng())
    assert out == "The committee did approve the budget."


def test_negation_insertion_plain_verb():
    out = _rule_negation(sent_of("The committee approved the budget."), rng())
    assert "never approved" in out


def test_negation_insertion_with_aux():
    out = _rule_negation(sent_of("The board will submit the report."), rng())
    assert "will not submit" in out


def test_negation_insertion_copula():
    out = _rule_negation(sent_of("The budget is large."), rng())
    assert "is not" in out


def test_negation_insertion_question_inversion():
    out = _rule_negation(sent_of("What do you think about the ban on large notes?"), rng())
    assert out == "What do you not think about the ban on large notes?"


def test_negation_insertion_copula_question():
    out = _rule_negation(sent_of("Is the budget large?"), rng())
    assert out == "Is the budget not large?"


def test_negation_skips_verb_initial():
    assert _rule_negation(sent_of("Submit the report."), rng()) is None


def test_number_digit():
    src = "The company hired 25 engineers in 2020."
    out = _rule_number(sent_of(src), rng())
    assert out is not None and out != src
    a, b = re.findall(r"\d+", src), re.findall(r"\d+", out)
    assert len(a) == len(b)
    assert sum(x != y for x, y in zip(a, b)) == 1, "exactly one number must change"


def test_number_word():
    out = _rule_number(sent_of("Two men entered the shop."), rng())
    assert out == "Three men entered the shop."


def test_arg_swap():
    out = _rule_arg_swap(sent_of("The cat chased the dog."), rng())
    assert out == "The dog chased the cat."


def test_entity_swap():
    out = _rule_entity_swap(sent_of("Alice met Bob in the lobby."), rng())
    assert out == "Bob met Alice in the lobby."


DOC3 = (
    "The committee approved the budget on Tuesday. "
    "The company hired 25 engineers in 2020. "
    "Alice met Bob in the lobby."
)


def test_multi_sentence_perturbs_exactly_one():
    sents = [s.text for s in _nlp()(DOC3).sents]
    pairs = generate_minimal_pairs(DOC3, n=4, seed=1)
    assert len(pairs) >= 3
    for p in pairs:
        assert isinstance(p, MinimalPair)
        assert p.text != DOC3
        untouched = [s for j, s in enumerate(sents) if j != p.sentence_idx]
        for s in untouched:
            assert s in p.text, f"sentence {s!r} must be byte-exact in the negative"
        assert sents[p.sentence_idx] not in p.text


def test_distinct_and_deterministic():
    a = generate_minimal_pairs(DOC3, n=4, seed=7)
    b = generate_minimal_pairs(DOC3, n=4, seed=7)
    assert [(p.text, p.rule, p.sentence_idx) for p in a] == [(p.text, p.rule, p.sentence_idx) for p in b]
    texts = [p.text for p in a]
    assert len(set(texts)) == len(texts)


def test_no_negatives_for_bare_fragment():
    # nothing parseable to perturb -> empty list, not garbage
    assert generate_minimal_pairs("Hello.", n=2, seed=0) == []
