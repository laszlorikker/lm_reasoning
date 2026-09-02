# DATA.md — datasets actually used (Phase 1 pilot)

Maintained per kickoff rules 7/8: every Hub dataset verified before use,
substitutions recorded, counts from the real build (runs/m1/pilot_stats.json).

## 1. Verification and substitutions (2026-08-31, `scripts/verify_datasets.py`)

Probed via streaming (no bulk downloads); full field/sample details in
`runs/m1/dataset_verification.json`.

| Planned source | Status | Resolution |
|---|---|---|
| paws-x (en/fr/de/es) | OK | used: positives as paraphrases, **label-0 rows as natural adversarial negatives** |
| glue/mrpc, glue/qqp | OK | used (en paraphrases) |
| opus-100 (de-en, en-es, en-fr, en-it, en-pt) | OK | used (sentence translation) |
| **tapaco** | UNAVAILABLE — datasets 5.x removed script-based loaders | **dropped**; multilingual paraphrase covered by paws-x languages + translation pairs |
| **tatoeba / tatoeba_mt** | UNAVAILABLE (same cause) | **dropped**; opus-100 covers sentence translation |
| xnli (all_languages) | OK | validation split used for the val pool (multilingual contradiction negatives); train (translationese MNLI) not used for training |
| multi_nli | OK | used: premises with both an entailment (weak positive) and a contradiction (hard negative) hypothesis |
| anli | OK | verified but unused in the pilot (reserved for eval-time hard negatives) |
| nvidia/OpenMathInstruct-2 | OK (`train_1M` split, streamed) | used: two solutions to one problem = two derivations of one conclusion |
| **facebook/babi_qa** | UNAVAILABLE (same cause) | **substituted: `Muennighoff/babi`** (parquet; fields passage/question/answer/task verified; tasks 15–16 filtered at M4, eval only) |
| europarl (5 configs) | OK | **row continuity verified** (consecutive rows form a continuous speech) → document-level source for concat-k |
| news_commentary (en-fr, de-en, en-es) | OK | document-ordered rows, same use |

## 2. Pilot corpus — `data/processed/pilot_v1.2` (recipe: `configs/base.yaml` → `data.pilot`, seed 17)

**v1.2 (2026-09-02, M3.1):** `math_derivation` rebuilt on `train_5M` (~6 GB,
explicitly approved) with the exact grouped join, cap 4 pairs/problem →
**9,771 pairs / 2.73M tokens** (K≥3 99.4%). The 30–50k target is **not
reachable from this split**: the "fair" subsets are diversity-sampled, so few
problems carry ≥2 guard-passing solutions even at 5M rows (2M gave 8,934).
**Decision (2026-09-02): accept ~10k for the pilot; no 14M download.** If the
M5 reasoning probe shows the propositional level weak, revisit as **v1.3**:
capped combination-pairing (each solution paired with ≤2 others,
near-identical solutions deduped). All other shards hardlinked from v1.1; same
dedup gate (17,268 removed). **Totals: 375,671 pairs, K≥3 33.4%, 44.8M
post-dedup src+tgt tokens.** Training uses v1.2.

**v1.1 (2026-09-01, M1 review):** rebuilt `qqp_singles` and `concat_paraphrase`
with the grammar-aware negation rule (aux inversion in questions, verb-initial
skip); `math_derivation` rebuilt with an exact group-by-problem join on
`train_2M` capped at 4 pairs/problem → **8,934 pairs** (the 30–50k target
needs `train_5M` at ~6 GB, pending an explicit download approval);
`concat_translation` kept from v1 (declarative text; a residual of ungrammatical
question-negatives estimated well under 0.1%). **Eval-leakage dedup gate**:
167,763 content hashes covering all fixtures and full held-out splits;
**17,268 pairs removed** (4.4% — mostly recurring qqp questions and split
overlaps), 16 negatives stripped. `pilot_v1` remains on disk for provenance;
all training uses v1.1.

Construction rules:
- Each example: `(source, target, lang_src, lang_tgt, pair_type, negatives[])`,
  tokenized with the Qwen3 tokenizer; **both sides chunked** to token spans
  (sentences via pysbd; `pt` uses the punctuation fallback — pysbd has no
  Portuguese rules). Caps: 512 tokens / 8 chunks / 64 tokens per chunk; pairs
  violating a cap after chunking are **dropped, not truncated** (counted below).
- Direction sampled 50/50 (en→xx vs xx→en; s1→s2 vs s2→s1).
- Multi-chunk pairs (M1 addition 2): `concat_translation` joins k∈[3,6]
  consecutive aligned europarl/news-commentary rows on both sides (terminal "."
  appended where a row lacks end punctuation — counted); `concat_paraphrase`
  concatenates k independent paraphrase pairs in the same order on both sides
  (synthetic discourse, chunk-aligned by construction); `math_derivation`
  solutions are naturally multi-sentence.
- Negatives: paws-x label-0 adversarials (all languages, attached to their own
  sentence1); MNLI contradiction hypotheses; **generated minimal pairs** on 25%
  of en-source examples (rules: negation flip, number perturbation,
  subject/object swap, same-label entity swap — `abstractnet/data/minimal_pairs.py`);
  for multi-sentence documents exactly ONE sentence is perturbed, the rest
  byte-exact (M1 addition 4); `concat_paraphrase` additionally substitutes one
  constituent with its adversarial partner where one exists.
- Identity targets are NOT in the data: `p_id` is applied at train time
  (collator, scheduled by the train loop).

### Measured (runs/m1/pilot_stats.json, build of 2026-08-31)

| source | pairs | tokens (src+tgt) | K≥3 | negatives attached |
|---|---:|---:|---:|---:|
| opus100_singles | 149,969 | 6,022,241 | 1.6% | 16,736 |
| concat_translation | 91,083 | 27,878,711 | 99.9% | 11,304 |
| paws_x_singles (en/fr/de/es) | 48,000 | 3,180,109 | 0.4% | 27,817 |
| qqp_singles | 40,000 | 968,483 | 0.5% | 9,881 |
| nli_entailment | 30,000 | 1,119,552 | 0.4% | 37,172 |
| concat_paraphrase | 21,642 | 3,220,036 | 99.8% | 8,394 |
| math_derivation | 5,105 | 1,414,037 | 99.5% | 1,249 |
| mrpc_singles | 2,474 | 131,167 | 0.3% | 579 |
| **total v1 (pre-dedup, superseded)** | **388,273** | **43,934,336** | **31.1%** | **113,132** |
| **total v1.1 (math regrouped 8,934 / +1.1M tok, then dedup −17,268)** | **374,834** | **45,000,094** | **33.2%** | — |

- **K≥3 share 31.1% of pairs — target ≥30% met** (M1 addition 2).
  K histogram: 1→245,962 · 2→21,673 · 3→24,224 · 4→27,011 · 5→27,380 ·
  6→26,016 · 7→11,307 · 8→4,700.
- Language-pair mix (top): en↔en 106,948; en↔{fr,es,de} ≈ 26.1–26.8k per
  direction; en↔{it,pt} ≈ 20.4–20.9k per direction; fr/de/es same-language
  ≈ 13.4k each (paws-x + concat_paraphrase).
- Negatives: 58,339 generated minimal pairs (spaCy rules, en), the rest
  natural (paws-x label-0 adversarials, MNLI contradictions, one-constituent
  substitutions in concat_paraphrase). 4 generated negatives dropped at caps.
- Drops (counted, not truncated): 14,590 source-over-cap, 4,400
  target-over-cap; 26,621 terminal periods appended in concat rows; 9,796
  concat groups skipped by guards.
- math_derivation undershot its 20k budget at **5,105 pairs**: in the wider
  train_5M stream, same-problem solutions sit farther apart, so the bounded
  100k-problem discovery buffer evicts entries before a partner arrives
  (train_1M had yielded 6,818 under the same buffer). Likely fix if more math
  mass is wanted later: larger buffer (RAM is plentiful) — deferred, no stated
  target depends on it.

### Example pairs (verbatim from data/processed/pilot_v1/full)

**nli_entailment** (multi_nli, en→en, K=1)
- SRC: "Dannie Abse told the London Observer , [Dylan's] writing is inferior poetry, and inferior poetry is not really poetry at all."
- TGT: "Dannie Abse talked to the London Observer."
- NEG (contradiction): "Dannie Abse did not spend any time talking to London Observer."

**concat_translation** (europarl-en-pt, en→pt, K=5)
- SRC: "In many cases they do not have the civic structures and other organisations in place to properly enforce some of the regulations we would like and some of the monitoring that would take place. So we cannot exclude one co…"
- TGT: "Em muitos casos não possuem estruturas civis e outras organizações aptas para impor o cumprimento de alguma da regulamentação…"

**paraphrase** (glue-qqp, en→en, K=1) — note the negative's grammar, see §4
- SRC: "What do you think about the ban on 500 and 1000 denomination notes in India?"
- TGT: "What are your views on demonetization of 500 and 1000 rupee notes by the Modi Government?"
- NEG (generated, negation): "What do not you think about the ban on 500 and 1000 denomination notes in India?"

**math_derivation** (openmath2-augmented_math, en→en, K=5)
- SRC: "To find the maximum number of boxes that can be stored in the container, we need to divide the volume of the container by the volume of a single box. First, calculate the volume of a single box: …"
- TGT: "To find the maximum number of boxes that can fit in the container, we need to calculate the volume of a single box and then divide the total volume of the container by the volume of a single box. …"

## 3. Frozen validation artifacts (committed under `data/fixtures/`)

Built by `scripts/build_val_panel.py`; separation from train is by
construction: held-out splits (paws-x test, qqp/mnli/xnli validation, opus-100
test), the first 3,000 documents per concat config (`VAL_RESERVED_DOCS`), and
math problems in the `is_val_problem` hash class (hash%97==0).

- **`val_pool_v1.jsonl`** — the fixed 2k-document subset for per-interval eval
  (VAL_REPORT_SPEC): text, language, k_est, paraphrase partner, reference
  translations, hard negatives with provenance.
- **`panel_v1.jsonl`** — the 32-document panel, ids `panel-00..31`, frozen
  forever: 12 multi-chunk documents (9 concat-translation from val-reserved
  regions with language-agnostic digit-bump negatives, 3 math derivations) +
  20 hand-written hard singles across en/fr/de/es/it/pt (negation-heavy,
  numbers, entities, long multi-clause; each with a controlled paraphrase, a
  hand-written minimal negative, and reference translations on several).

Measured (runs/m1/val_panel_stats.json; **v2 frozen 2026-09-01** — v1 was
en-heavy with it/pt at 50 because a per-config budgeting bug drew all concat
docs from europarl-en-fr; v1 deleted, no checkpoint was ever evaluated on it):

- **val_pool_v2: exactly 2,000 docs.** Languages: en 744, de 295, es 295,
  fr 292, **it 187, pt 187** (requirement: ≥150 per non-English language).
  Origins: opus-100 held-out 608, paws-x test 480, europarl val-reserved 308,
  xnli validation 200, news-commentary val-reserved 184, multi_nli validation
  160, openmath2 val-hash 60. K≥3: 567 (28.4%). Coverage: 900 with a
  paraphrase partner, 1,100 with a translation partner, 885 with at least one
  hard negative.
- **Minimal-pair audit (M1.1 gate b)** — chrF(x, x⁻) per rule on a 2k-source
  seeded sample; high chrF = surface-close (the point of a minimal pair);
  20 samples per rule in `data/fixtures/minimal_pair_samples.md`:

  | rule | n | chrF mean | p10 | min |
  |---|---:|---:|---:|---:|
  | negation | 2,775 | 95.7 | 88.4 | 32.7 |
  | number | 644 | 95.9 | 89.8 | 32.7 |
  | entity_swap | 305 | 94.5 | 88.5 | 68.9 |
  | arg_swap | 976 | 91.6 | 75.7 | 36.4 |
- **panel_v1: 32 docs** (`panel-00..panel-31`): 12 multi-chunk (9 concat
  translation + 3 math derivations) + 20 hand-written hard singles. Languages:
  en 15, de 5, fr 4, it 3, pt 3, es 2 — all six covered. 23 with paraphrase,
  14 with reference translations, 30 with hard negatives (2 drawn docs had no
  applicable language-agnostic rule; the failure tables draw from the pool, so
  coverage is unaffected).

## 4. Known v1 limitations

- The minimal-pair *generator* is English-only; multilingual hard negatives are
  natural (paws-x label-0, xnli contradictions) plus the language-agnostic
  digit-bump used in the panel. LLM-generated rewrites: hook in
  `minimal_pairs.GENERATORS`, unimplemented by design.
- **Negation insertion on interrogatives is ungrammatical** ("What do not you
  think…"): meaning still flips, and the contrastive term does not score
  fluency, but a fluency-sensitive consumer should filter by rule. Affects the
  `negation` rule on question-shaped sources (qqp) only.
- **paws-x TRAIN sides are machine-translated** for fr/de/es (test is human
  translated) — visible as noisy phrasing in some paraphrase/concat_paraphrase
  rows. Kept: the equivalence classes and adversarial negatives remain valid;
  the val pool draws from the human-translated test split.
- `concat_paraphrase` documents are synthetic discourse (independent sentences).
- Portuguese sentence segmentation uses the punctuation fallback.
- xnli train and anli are available but unused for training in the pilot.
