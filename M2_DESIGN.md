# M2 design note — AbstractLM

The M2 kickoff spec (2026-09-01) is binding; this file records the readings
taken where it left a choice, and every deviation. Review at M2 acceptance.

## Readings of the spec

1. **Pooling MLP width** — "MLP (h → 4h → d_z)": h read as the attention output
   width (n_heads·head_dim = 1024), giving 1024 → 4096 → 1024. Reading the
   model hidden (2048) would double the pooler for no obvious gain.
2. **z projection and chunk-position table are SHARED across blocks** (the
   trainable-set list names one "z projection"); each block owns its LN,
   q/k/v/o projections and scalar zero-init gate. z_h = z_proj(z) + chunk_pos
   is computed once per forward in AbstractLM and handed to all 7 wrappers
   (split=14, every 2 → layers 14,16,18,20,22,24,26).
3. **"LN" = nn.LayerNorm** in new modules (autocast computes it in fp32); the
   base model's RMSNorm stays untouched inside the wrapped layers.
4. **Language-token loss masking**: implemented as hidden[:, :-1] scoring
   labels[0..T) — no position is asked to predict the lang token (it is given),
   while the lang-token position itself still predicts the first target token
   from z. "Its position is masked from the loss" read as the former; the
   first-token prediction is essential, without it the decoder gets a free
   start.
5. **attention_mask=None everywhere**: with causal attention and right padding,
   real tokens never attend pad positions (pads lie in the future); pad states
   are never read (pooling masks spans, CE ignores -100). This uses the
   model's own mask construction path exactly, per spec §1.
6. **Rate applies to z(x) only** — the channel the decoder actually reads.
   z(x′), z(x⁻) and the contrastive term use clean (pre-noise / μ) vectors.
7. **encode() takes optional `langs`** (default en): chunking needs the
   language; §8's signature is otherwise unchanged.
8. **decode_logits returns token log-probs + per-doc NLL**, not full-vocab
   logits (a 512-token doc's logits are 0.3 GiB against 151k vocab; scoring
   needs neither).
9. **encoder_lora_layers** exists in config, default empty; a non-empty value
   raises NotImplementedError until the M5 ablation wires the named adapter.
   With it empty the lower half contains no adapters at all (asserted at init),
   so bit-identity with the base LM is structural.
10. **Identity rows (p_id)** make the contrastive positive trivial for those
    rows (z(x) vs z(x)); accepted — p_id is small and decaying.

## Implementation constraints worth knowing

- **xattn z-context lifetime**: wrappers read their z at BACKWARD time under
  gradient checkpointing (non-reentrant recompute). set_z overwrites per
  forward and is deliberately not cleared after training steps; clear_z() is
  for base-equality / no-z paths only.
- Trainable params are fp32 (LoRA tensors explicitly upcast); base stays fp16;
  all forward paths run under fp16 autocast.
- The three encoder groups (x, x′, x⁻) are padded to a common (S, K) and run
  as ONE lower-half forward, then sliced back to their own K (spec §5
  "batched together").
- Word dropout: Bernoulli over real target positions only, replaces the input
  embedding with one learned vector; labels and the lang position untouched.
- decode() is a hand-rolled greedy loop over `model(inputs_embeds=…,
  past_key_values=…)`; cross-attention is cache-independent (keys are z).
- **Encoder lower half runs under no_grad** (frozen, no adapters below split in
  M2/M3): gradients reach the pooler and stop there by construction — exact,
  and the lower-half activations never exist in the train step. When the M5
  encoder-LoRA ablation lands, this becomes conditional on the adapter config.

## Deviations from the spec

None at module level. Deferred inside it: encoder_lora_layers named-adapter
creation (M5), sampling in decode() (greedy only, per the val-report decision).
