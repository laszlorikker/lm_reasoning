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

## 2. Pilot corpus — `data/processed/pilot_v1` (recipe: `configs/base.yaml` → `data.pilot`, seed 17)

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

### Measured (runs/m1/pilot_stats.json)

TBD_PILOT_TABLE

TBD_PILOT_AGGREGATE

### Example pairs

TBD_EXAMPLES

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

TBD_VAL_PANEL_STATS

## 4. Known v1 limitations

- The minimal-pair *generator* is English-only; multilingual hard negatives are
  natural (paws-x label-0, xnli contradictions) plus the language-agnostic
  digit-bump used in the panel. LLM-generated rewrites: hook in
  `minimal_pairs.GENERATORS`, unimplemented by design.
- `concat_paraphrase` documents are synthetic discourse (independent sentences).
- Portuguese sentence segmentation uses the punctuation fallback.
- xnli train and anli are available but unused for training in the pilot.
