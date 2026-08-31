# Phase 1 — Abstraction Net (Encoder / Decoder) on Qwen3-1.7B-Base

## 0. Scope

This project builds a latent reasoning system in three parts: an **abstraction-net encoder** (language → sequence of abstraction vectors), a **reasoning model** operating in that latent space, and an **abstraction-net decoder** (latent → language).

**Phase 1 (this document):** turn a pretrained LM into the encoder/decoder pair. Train it so the bottleneck carries *meaning, not surface form*, and ship a clean latent interface (`encode` / `decode`) for Phase 2.

**Phase 2 (later, out of scope here):** design and train the recurrent reasoning model inside the bottleneck. Nothing in Phase 1 should assume a particular reasoner. Do not build one.

Guiding principle: the LM is the *interface*, not the reasoner. Brute-force pretraining already paid for the arbitrary part (language ↔ meaning). We reuse that and never train the LM on next-token prediction of its own input.

## 1. Hardware and its consequences

Single **NVIDIA Quadro RTX 5000, 16 GB, Turing (sm_75)**. This dictates:

- **No bf16.** Use fp16 autocast with `GradScaler`; keep master weights of all trainable modules in fp32. Watch for overflow (norm layers already run in fp32 in the HF Qwen3 implementation).
- **No FlashAttention-2** (needs sm_80+). Do **not** install `flash-attn`. Load the model with `attn_implementation="sdpa"` and let PyTorch pick the memory-efficient kernel.
- **bitsandbytes works on Turing**: use `AdamW8bit` and, for the 4B upgrade, 4-bit base weights.
- Triton-based kernels (Liger, flash-linear-attention, etc.) are hit-and-miss on sm_75. Avoid them unless proven in `M0`.
- Verify at setup: `torch.cuda.get_device_capability() == (7, 5)` and that the installed PyTorch wheel includes sm_75.
- **Memory budget: ≤ 14 GB peak** during training, so runs survive fragmentation. Log peak memory every eval.

## 2. Model

**`Qwen/Qwen3-1.7B-Base`** (Apache 2.0). Base checkpoint, never Instruct. Expected: 28 layers, hidden 2048, 16 query / 8 KV heads, ~151k vocab, tied embeddings — **verify all of this from `config.json` after pulling; do not trust these numbers.**

Upgrade path (not now): `Qwen/Qwen3-4B-Base` (36 layers) with 4-bit base weights. Keep every layer index configurable so the upgrade is a config change.

### 2.1 Architecture

One LM, split at the middle. `SPLIT = 14` by default (configurable; try 12 / 14 / 16 in ablation).

**Encoder**
1. Run layers `0..SPLIT-1` on the source text (frozen).
2. Take the residual stream after layer `SPLIT-1` as token states.
3. **Chunking:** v1 chunks are sentences (use `pysbd` or spaCy sentencizer; split any sentence longer than 64 tokens; cap documents at 8 chunks / 512 tokens). Chunk boundaries are precomputed in the data pipeline and passed in as token spans. Learned segmentation is a later ablation, not v1.
4. **Pooling head** (trainable, new): per chunk, single-query attention pooling over that chunk's token states (masked to the span) → LayerNorm → 2-layer MLP → `z_k ∈ R^{d_z}`, `d_z = 1024`.
5. **Rate term** on `z` (configurable): `noise` (add N(0, σ²), σ=0.1, train only) or `kl` (VIB: predict μ, logσ; KL to N(0, I), β=1e-3). Default `noise`.
6. `z` carries **no positional information**; chunk order is supplied on the decoder side.

Output contract: `z: [B, K, d_z]`, `z_mask: [B, K]`, `chunk_spans`.

**Decoder**
1. The full LM stack runs on the *target* text (teacher forcing). The lower half is the same frozen weights as the encoder's lower half (shared).
2. **Gated cross-attention blocks** (new, trainable) inserted before decoder layers `SPLIT..L-1`, every 2nd layer by default (configurable). Each block: LN → multi-head attention with queries from the hidden state, keys/values from `z` projected to the hidden size plus a **learned chunk-position embedding** on the keys → `tanh(gate) * out` added to the residual. **Gates initialised to zero**, so at step 0 the decoder is exactly the base LM.
3. **Target-language conditioning:** a learned soft token per target language (`en`, `fr`, `de`, `es`, …) prepended to the decoder input. Required for cross-lingual reconstruction and eval.
4. **Word dropout** on decoder input tokens (replace 20% with a reserved token, train only, configurable) so the decoder cannot copy from its own prefix and must read `z`.
5. LM head stays frozen (tied to embeddings).

**Trainable parameters:** pooling head, cross-attention blocks and gates, language soft tokens, and **LoRA (r=16, α=32)** on q/k/v/o and MLP projections of layers `SPLIT..L-1`. Everything else frozen. Report trainable-parameter count in `M2`.

## 3. Data

The objective needs *equivalence classes*: pairs `(x, x')` that mean the same thing but differ in surface form, plus **minimal-pair negatives** `(x, x⁻)` that differ in exactly one proposition.

v1 sources (verify each on the Hub; substitute equivalents if missing; record what was used in `DATA.md`):

| Role | Candidates |
|---|---|
| Paraphrase, with adversarial negatives | `paws-x` (all languages), `glue/mrpc`, `glue/qqp` |
| Translation pairs | `opus-100` (en↔fr/de/es/it/pt to start), Tatoeba |
| Multilingual paraphrase | `tapaco` |
| Entailment (weak positives) / contradiction (hard negatives) | `xnli`, `multi_nli`, `anli` |
| Multiple derivations of one conclusion | `nvidia/OpenMathInstruct-2` (several solutions per problem) |
| Reasoning probe tasks (eval only) | `facebook/babi_qa` tasks 15–16 guaranteed; add ProofWriter / ProntoQA if located |

**Minimal-pair generator** (`data/make_minimal_pairs.py`), rule-based for v1: negation insertion/removal, number perturbation, argument swap around transitive verbs (dependency parse), entity swap via NER. LLM-generated rewrites and negatives are a later addition; leave a hook.

Pair construction: each training example is `(source, target, lang_target, negatives[])`. With probability `p_id` (0.2 for the first 20% of steps, decaying to 0.05) the target is the source itself, to stabilise early training. Sequence caps: 512 tokens source, 512 target.

Pilot size: **~50M tokens** of pairs for the first real run. Do not build a billion-token pipeline in Phase 1.

## 4. Losses

`L = L_recon + λ_c · L_contrastive + λ_r · L_rate`

- **`L_recon`** — cross-entropy of the decoder on the target given `z` (cross-paraphrase / cross-lingual reconstruction). This is the main term. Compute the LM-head cross-entropy in **chunks** (e.g. 1024 tokens at a time) — ~151k-vocab logits are the biggest single memory cost on 16 GB.
- **`L_contrastive`** — InfoNCE (τ=0.05, symmetric) on **document vectors** `mean_k(z_k)`: positives are paraphrase/translation partners, negatives are in-batch plus the minimal pairs. Document-level for v1 because paraphrases do not chunk-align. `λ_c = 0.1`.
- **`L_rate`** — KL term if `rate=kl`; zero if `rate=noise`. `λ_r = 1e-3`.

Formal-language head (Lean / code from `z`) is **Phase 1b**: reserve a config flag and an output head slot; do not implement in the first pass.

## 5. Training setup

- Framework: PyTorch + `transformers` + `peft` + `bitsandbytes` + `datasets` + `accelerate`. Pin versions in `requirements.txt`; confirm the CUDA wheel supports sm_75.
- Optimiser: `AdamW8bit`. LR 2e-4 for new modules, 1e-4 for LoRA, cosine schedule, 500 warmup steps, weight decay 0.01, grad clip 1.0.
- Precision: fp16 autocast + `GradScaler`; fp32 master weights for trainable modules.
- Gradient checkpointing on the decoder half. Micro-batch 8 pairs × 512 tokens, gradient accumulation to an effective batch of 64. Adjust from the `M0` memory profile.
- Logging: local TensorBoard (or W&B if configured). Log per-loss terms, gate magnitudes per layer, `z` statistics (mean norm, per-dim variance — collapse detector), tokens/s, peak memory.
- Checkpointing: save trainable weights only (LoRA + new modules), plus a full `save_pretrained`-style bundle at milestones.
- Run-length policy: no run longer than 30 minutes without an explicit go-ahead.

## 6. Evaluation — the acceptance tests

All in `eval/`, each runnable standalone on a checkpoint, all results appended to `RESULTS.md`.

1. **Invariance AUC.** Cosine similarity in `z`-space (document vectors) between paraphrase pairs vs minimal-pair / adversarial negatives (`paws-x` test, `xnli` contradiction pairs). Report AUC. Baseline: same metric on mean-pooled frozen layer-`SPLIT` states and on frozen final-layer states.
2. **Meaning preservation of reconstruction.** Bidirectional NLI (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` or `microsoft/deberta-large-mnli` for English) between source and decoded output → entailment both ways. Also chrF and BERTScore. **BLEU is reported but is not a target** — a good abstraction reconstructs meaning, not strings.
3. **Cross-lingual.** Encode in one language, decode into another with the language soft token; score with chrF and COMET (`Unbabel/wmt22-comet-da`) against reference translations.
4. **z-dependence (z-swap).** Decode with the correct `z`, with `z` from a different document, and with `z` zeroed. Reconstruction quality must fall sharply when `z` is wrong. This is the test that the decoder is not just a fluent LM ignoring the bottleneck.
5. **Reasoning probe.** Freeze the encoder. Train a small transformer (2 layers) on `z` sequences for bAbI 15–16 (and ProofWriter / ProntoQA if available). Compare with the *same* probe on frozen layer-`SPLIT` mean-pooled states and on frozen final-layer states. `z` must beat both, or the ladder bought nothing.

Also report: compression ratio (tokens per chunk), gate magnitudes, `z` variance spectrum.

## 7. Milestones

Stop at the end of each milestone, write a short report, and wait for review.

- **M0 — Environment and smoke test.** Inspect GPU/driver/CUDA; create venv; pin deps; pull `Qwen3-1.7B-Base`; verify config; load with `sdpa` in fp16; forward pass at 512 tokens; **memory profile** for the split forward (encoder half + decoder full stack, batch 8). Write `CLAUDE.md` with conventions and the hardware constraints.
- **M1 — Data pipeline.** Download and cache datasets; chunking; pair construction; minimal-pair generator; collator; `DATA.md` with counts, language mix, and example pairs. Unit tests on chunking and collation.
- **M2 — Model.** `AbstractLM` module: split, pooling head, gated cross-attention, language tokens, LoRA wiring, rate term. Tests: shapes; with gates at zero, decoder logits equal the base LM's; trainable-parameter count; memory profile of a full train step.
- **M3 — Training.** `train.py` + YAML configs; 1k-step pilot; loss curves; gate growth; `z` statistics. Confirm no collapse and no fp16 overflow.
- **M4 — Evaluation suite.** All five acceptance tests running on the M3 checkpoint, with baselines.
- **M5 — Pilot run.** ~50M tokens. Results table in `RESULTS.md`. Ablations if time: `SPLIT ∈ {12,14,16}`, `rate ∈ {noise, kl}`, word dropout on/off. Decision point on the 4B upgrade.

## 8. Phase 2 interface (what Phase 1 must leave behind)

```python
model.encode(texts: list[str]) -> (z: Tensor[B,K,d_z], z_mask: Tensor[B,K], spans)
model.decode(z, z_mask, lang: str, max_new_tokens=256) -> list[str]
model.decode_logits(z, z_mask, lang, target_ids)  # teacher-forced scoring
```

Plus: a saved checkpoint loadable in one call; `z` statistics (mean, covariance) for Phase 2 initialisation; a script to dump a `z`-dataset for Phase 2 experiments. The reasoner will read and write `z` in this space — it is the only coupling between phases.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Decoder ignores `z` (fluent LM prior) | Paraphrase targets, word dropout, z-swap test as a gate for every milestone |
| `z` captures gist only, not propositions | Minimal-pair negatives in the contrastive term; invariance AUC on adversarial pairs; Phase 1b formal head |
| Representation collapse | Rate term, per-dim variance logging, contrastive term |
| fp16 overflow / NaN | `GradScaler`, fp32 masters, grad clip, loss-scale logging |
| OOM on 16 GB | Chunked CE, checkpointing, sequence caps, `M0` profile before any training |
| Turing kernel incompatibilities | `sdpa` only, no flash-attn, no Triton kernels unless proven in `M0` |
| Paraphrase chunks don't align | Document-level contrastive for v1 |
