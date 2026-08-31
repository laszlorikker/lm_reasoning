# Validation report — spec (addition to M3/M4, confirmed 2026-08-31)

One script, `eval/val_report.py`, run at every eval interval during training and
reusable standalone on any checkpoint. Output per eval:
`runs/<run>/val/step_XXXXXX/` containing a **self-contained HTML page**
(matplotlib PNGs embedded as data URIs) plus the same scalars/images written to
TensorBoard.

## Graphs (each overlaying all previous eval steps where that makes sense)

1. **Loss terms** — recon / contrastive / rate, train and val, plus GradScaler
   loss scale (fp16 health).
2. **Cross-attention gate magnitude per layer** over steps — is the decoder
   learning to read `z`?
3. **z variance spectrum** (sorted per-dim variance) and **effective rank** over
   steps — collapse detector.
4. **Invariance** — overlaid histograms of z-cosine for paraphrase pairs, hard
   negatives, random negatives; AUC over steps.
5. **z-dependence** — reconstruction score (bidirectional NLI, chrF) for
   correct z vs swapped z vs zeroed z, over steps.
6. **PCA of document z vectors**, coloured by language and (separately) by
   source document. Translation pairs should overlap; if z clusters by
   language, invariance has failed.
7. **Cross-lingual chrF/COMET** as a source×target language heatmap.
8. **Throughput and peak memory over wall-clock time**, with GPU clock and
   temperature on a secondary axis.

## Examples panel

A fixed panel of **32 validation documents** — seeded, spanning all languages in
the mix, the *same* panel every eval so steps can be diffed. Per document:

- source with chunk boundaries marked;
- target paraphrase;
- decoded output with correct `z`, with swapped `z`, with zeroed `z`;
- NLI scores for each decode;
- top-3 nearest neighbours in z-space from the val pool, with cosines.

Cross-lingual rows: encode in one language, decode into two others.

Two failure tables:

- the 10 lowest-NLI reconstructions;
- the 10 hard negatives with the highest cosine to their source (where
  minimality is being missed).

## Cost budget

- Panel and histograms come from a **fixed 2k-document val subset**; full
  metrics only at milestone evals. Target: **report renders in under two
  minutes**.

## Implementation notes (mine — review at M3)

- **Per-interval vs milestone split** (from "full metrics only at milestone
  evals"): every interval → graphs 1, 2, 3, 8, invariance histograms/AUC (4),
  z-dependence on the panel (5), PCA (6), examples panel, failure tables;
  milestone evals only → COMET and the full source×target heatmap (7) and any
  full-test-set metric. chrF (sacrebleu) is cheap enough for every interval;
  COMET is not.
- **Two-minute budget levers**: encode the 2k pool in one batched fp16 pass
  (encoder half only where possible); batch all panel generation (32 docs × 3
  z-variants in few batches, `max_new_tokens` a config knob); NLI model is
  small (mDeBERTa-base class) — load per eval or keep resident, decide from the
  M3 memory profile; never load COMET during training evals.
- **Over-steps overlays need history**: append per-interval scalars to
  `runs/<run>/val/history.jsonl`; the plotter reads it. Standalone invocations
  on a checkpoint append to (or create) the same file keyed by step.
- **Graph 8 data source**: train.py logs a telemetry CSV (sampler pattern
  already in `scripts/bench_sustained.py`) — wall-clock, tokens/s, peak
  alloc/reserved, SM clock, power, temperature per log interval.
- **M1 coupling**: the data pipeline must reserve (a) the fixed 2k-document val
  subset with paraphrase partners, hard negatives, and translation pairs with
  language labels, and (b) the seeded 32-document panel spanning the language
  mix — both frozen artifacts with stable IDs, so every eval and every
  checkpoint sees the identical panel.
- New deps when this lands (M3): matplotlib, sacrebleu; NLI checkpoint per
  PHASE1_PLAN §6; COMET (M4, milestone evals only).
