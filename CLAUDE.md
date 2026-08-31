# CLAUDE.md — abstractnet, Phase 1

`PHASE1_PLAN.md` is the spec and the authority. This file: what is true about this
box, the working conventions, and how to run things.

## Hardware truth (measured here, 2026-08-31)

- GPU: Quadro RTX 5000 **with Max-Q Design** — laptop TU104, 16 GiB, **sm_75**,
  power cap **85 W** (desktop RTX 5000 is 230 W: expect roughly half its
  throughput; all §1 constraints unchanged).
- WSL2 (kernel 5.15), Windows driver 553.35 → CUDA **12.4** userspace. Display
  runs on the iGPU: Windows-side baseline VRAM usage measured at **23 MiB**, so
  the plan's **14 GiB peak budget stands** (`profile.vram_budget_gib`, gated on
  peak *reserved*).
- CUDA context + WSL overhead ≈ **1.09 GiB** at torch init (driver reported
  14.91/16.00 GiB free before our first allocation). 14 GiB reserved budget ⇒
  ~0.9 GiB true device slack. Do not raise the budget.
- **fp16 only** (sm_75 has no bf16) with `GradScaler`; fp32 master weights for
  trainable modules. **SDPA only** — no flash-attn, no Triton kernels.
  Forced mem-efficient SDPA verified fwd+bwd at head_dim 128 (scripts/check_env.py).
  `enable_gqa` is unsupported on this build → transformers falls back to
  `repeat_kv` (correct; KV expansion 8→16 heads costs a little extra memory).
- bitsandbytes 0.50.2 `AdamW8bit` verified with a real 8-bit step on this card.
- transformers **v5** (5.16.1): rope config lives in `config.rope_parameters`
  (not `rope_theta`); `from_pretrained(dtype=...)` (not `torch_dtype`).
- transformers v5 returns **fp16** logits from forward (no fp32 upcast — verified).
  The fp32 cost lives in the loss: full-sequence fp32 cross-entropy adds
  ~4.6 GiB at 8×512×151936. Training loss must be chunked CE over *flattened*
  positions with per-chunk recompute — `chunked_ce(..., use_checkpoint=True)`
  in scripts/profile_memory.py, measured at +0.3 GiB (PHASE1_PLAN §4).
- **WSL2 does not OOM — it oversubscribes.** Past-16-GiB allocations silently
  spill to host RAM: measured 18.2 GiB "allocated" completing ~9× slower
  (21.9 s vs 2.4 s per step). Overflow here is a silent slowdown, not a crash.
  The profile-script budget gate is the real protection, and M3's train.py must
  call `torch.cuda.set_per_process_memory_fraction(budget/16)` so a genuine OOM
  error fires instead of a spill.

## M0 measurements

- Model: Qwen3-1.7B-Base, verified from config.json: 28 layers, hidden 2048,
  16 Q / 8 KV heads, head_dim 128, intermediate 6144, vocab 151936, tied
  embeddings, max_pos 32768, rope_theta 1e6. RMSNorm upcasts to fp32 internally.
  1.721 B params → **3.21 GiB** resident in fp16.
- Pure forward B=8 S=512: peak 4.39 GiB alloc / 4.94 GiB reserved (fp16 logits).
- Throughput, forward B=8 S=512 (10-min sustained bench): **burst 5,530 tok/s**
  (minute 1: 1125 MHz SM, 85 W, 65 °C) → **sustained 4,308 tok/s** (minutes
  2–10 mean; floor 3,618 tok/s once soaked at 79–80 °C, 600–750 MHz SM).
  Size every run estimate from the SUSTAINED number, never the burst number.
  Raw telemetry: runs/m0/ (regenerate with scripts/bench_sustained.py).
- Split-forward memory envelope (B=8 S=512, GiB peak reserved): weights-only
  3.21 alloc · encoder half fwd 3.78 · full-stack fwd 4.94 · chunked-CE fwd
  5.23 · **split-forward envelope 4.94 (budget 14.0 — WITHIN)** · train-shaped
  fwd+bwd preview: 6.22 with layer checkpointing + chunked CE (2.4 s) vs 18.65
  without checkpointing (spills past 16 GiB, 21.9 s). Layer checkpointing and
  chunk-recomputed CE are both mandatory in M2/M3.

## Environment

- conda env **`abstractnet`** (Python 3.12.14):
  `conda activate abstractnet`, or call
  `/home/laszlo/miniconda3/envs/abstractnet/bin/python` directly.
- Exact pins: `requirements.txt` — the versions that passed the gates. torch is
  `2.10.0+cu126` from the cu126 index; it runs on the 12.4 driver via CUDA
  minor-version compatibility, verified empirically.
- After ANY environment change, re-run: `python scripts/check_env.py --with-bnb`.
- HF cache: default `~/.cache/huggingface` (1 TB ext4 disk). Keep data on the
  Linux filesystem, never under /mnt/c.

## Run policy

- **≤ 30 minutes per run through M4.** Ask before anything longer.
- M3 must ship checkpointing every ≤ 15 minutes plus a *tested*
  resume-from-checkpoint, so M5 can run unattended in 30-minute windows.
- Every milestone that touches the model ends with
  `python scripts/profile_memory.py` at the configured batch/seq — it
  hard-fails when peak reserved exceeds the budget (kickoff rule 4).
- Log peak memory at every eval; watch the fp16 loss scale in training logs.

## Conventions

- Plain, readable PyTorch. No trainer frameworks; `transformers` is for
  loading the base model/tokenizer, `peft` for LoRA wiring only.
- Every experimental knob lives in `configs/*.yaml`, loaded by
  `abstractnet/config.py` (strict: unknown keys are errors). The Qwen3-4B
  upgrade must stay a pure config change (`configs/qwen3_4b.yaml`).
- **The non-negotiable objective:** the decoder is trained on paraphrases /
  translations of what the encoder saw — never on reproducing the encoder's
  input string. Identity targets exist only through the decaying `p_id`
  schedule. Any change that amounts to next-token prediction of the model's own
  input is the failure mode this project exists to avoid: stop and rethink.
- Phase 2 touches this code only through `encode` / `decode` / `decode_logits`
  (PHASE1_PLAN §8). Nothing in Phase 1 may assume a particular reasoner.
- Tests live in `tests/` (pytest). Mandatory before any training: with
  cross-attention gates at zero, decoder logits must equal the base LM's.
- Docs update as work happens, not at milestone end: `DATA.md` (datasets
  actually used, counts, language mix, examples), `RESULTS.md` (one table per
  evaluation, with baselines).
- Milestones end with a short report and a stop for review. No unasked starts.

## How to run

```bash
conda activate abstractnet
python scripts/check_env.py --with-bnb     # environment gates
python scripts/m0_smoke.py                 # config verify + load + forward smoke
python scripts/bench_sustained.py          # 10-min sustained throughput + telemetry
python scripts/profile_memory.py           # memory envelope vs budget (hard gate)
```

All scripts accept `--config configs/base.yaml` (the default).
