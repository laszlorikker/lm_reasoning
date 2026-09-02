# RUNBOOK — operating the Phase-1 pilot training

Written for the operator. Every command is copy-pasteable and runs from the
repo root. `<run>` is your run name, e.g. `pilot1`.

## 0. Environment and directory

```bash
cd ~/reasoning_lm
conda activate abstractnet          # env name is exactly "abstractnet"
```

Every command below assumes this env and this directory.

## 1. Pre-flight check (before any run)

```bash
nvidia-smi                                          # GPU visible, memory ~23 MiB used when idle
df -h ~ | tail -1                                   # >= 50 GB free
python scripts/check_env.py --with-bnb              # all gates PASS
python -m abstractnet.train --config configs/base.yaml --run runs/preflight --dry-run
```

The dry run validates the corpus schema, collates sample batches, and prints
steps per epoch. Expect "DRY RUN OK". The memory guard
(`torch.cuda.set_per_process_memory_fraction`) is armed automatically inside
every training process; on this WSL2 box an over-budget step then raises a loud
OOM instead of silently spilling to host RAM at ~9× slowdown.

Optional hardening (Windows side): NVIDIA Control Panel → Manage 3D settings →
"CUDA - Sysmem Fallback Policy" → "Prefer No Sysmem Fallback". Not required —
the in-process guard already covers training.

## 2. Starting the pilot

**Attended, in a 30-minute window** (the M0–M4 policy):

```bash
tmux new -s pilot
python -m abstractnet.train --config configs/base.yaml --run runs/pilot1 --max-wall-minutes 30
```

It saves and exits 0 when the window elapses. Start the next window with the
resume command below.

**Unattended overnight** (only after you've done at least one attended window
and the resume drill):

```bash
tmux new -s pilot
python -m abstractnet.train --config configs/base.yaml --run runs/pilot1 --max-wall-minutes 0
```

The difference: `--max-wall-minutes 0` means no wall-clock exit — the run goes
until the corpus pass completes, you stop it, or something breaks. Checkpoints
land every 15 minutes / 500 steps either way, so the most you can lose is
15 minutes of work.

Detach from tmux with `Ctrl-b d`; reattach with `tmux attach -t pilot`.

## 3. Resume / stop

```bash
# resume from the newest valid checkpoint in the run dir:
python -m abstractnet.train --config configs/base.yaml --run runs/pilot1 --resume auto --max-wall-minutes 30

# resume from a specific checkpoint:
python -m abstractnet.train --config configs/base.yaml --run runs/pilot1 --resume runs/pilot1/step_004500.pt

# stop cleanly from inside the tmux pane:  Ctrl-C   (finishes the step, saves, exits 0)
# stop from any other terminal:
touch runs/pilot1/STOP                    # picked up at the next step; file is removed on exit
```

Resume refuses to start if `configs/base.yaml` changed since the checkpoint —
that's deliberate. If the change is intended, add `--allow-config-change`.

**After a crash, a Windows sleep, or a reboot:** just run the `--resume auto`
command. The checkpoint format is atomic — a kill mid-save cannot corrupt it;
resume picks the newest checkpoint that loads. The resume proof
(`scripts/test_resume.py`) verified continuation is step-exact: same batches,
same LRs, same optimiser step counts (numbers in the M3 report).

## 4. Monitoring

```bash
tmux attach -t pilot                      # the live log
tensorboard --logdir runs/pilot1/tb --port 6006
# then open in the Windows browser:  http://localhost:6006
```

**First 10 minutes — healthy looks like:**
- `loss/recon` falling steadily; `loss/contrastive` falling from ~2.0
- `opt/loss_scale` stable after settling (thousands; it halves on overflow —
  occasional halvings are fine, a downward staircase is not)
- `gates/layer_*` drifting off zero and growing
- `z/dim_var_mean` not collapsing toward zero (collapse detector)
- `perf/pair_tokens_s` steady around the expected number (§7), GPU SM clock in
  the telemetry file settling near ~600–750 MHz once heat-soaked (that is
  normal Max-Q behaviour, not a fault)

**Red flags — stop now (Ctrl-C or `touch runs/pilot1/STOP`):**
- any NaN in the losses, or `opt/loss_scale` collapsing below ~128
- `perf/peak_alloc_gib` climbing toward 14, or an OOM message (the guard fired)
- tokens/s cliff (drops by half and stays) — check telemetry for temperature
  pinned at ~87 °C or clocks stuck at minimum; also check Windows didn't start
  something heavy on the GPU
- steps.jsonl `overflows` counter growing every few steps

**Telemetry file:** `runs/pilot1/telemetry.csv` — one row per 30 s:
`t_unix, sm_clock_mhz, power_w, temp_c, mem_used_mib`. Cross-reference with
`perf/*` in TensorBoard; the val report plots them together (graph 8).

## 5. Validation reports

During training a report is generated automatically (interval auto-set so it
costs < 10% of training time). On any checkpoint, standalone:

```bash
python -m eval.val_report --checkpoint runs/pilot1/step_004500.pt --run runs/pilot1
```

HTML lands at `runs/pilot1/val/step_XXXXXX/report.html` — open it from Windows
at `\\wsl$\Ubuntu\home\laszlo\reasoning_lm\runs\pilot1\val\...` or serve it:
`python -m http.server 8000` then `http://localhost:8000/runs/...`.

What to read in it: AUC (hard) trending up; the NLI gap correct-vs-swapped/zeroed
opening (that gap IS the z-dependence result); effective rank not collapsing;
the panel decodes becoming paraphrases rather than copies or babble.

## 6. WSL2 specifics

- **Always run inside `tmux`** — closing the terminal window kills any
  foreground process. `tmux ls` shows sessions; `tmux attach -t pilot` returns.
- **Windows must not sleep** while plugged in: Settings → System → Power →
  "When plugged in, put my device to sleep" → **Never**. A sleep suspends the
  GPU mid-step; the run usually dies. If it happens: resume as in §3.
- Check liveness from a new terminal:
  `ps aux | grep abstractnet.train | grep -v grep` and
  `tail -2 runs/pilot1/steps.jsonl` (the step number should advance).
- Keep everything on the Linux filesystem (it already is); never move the run
  dir to /mnt/c.

## 7. Expected numbers (measured in M3.1, 2026-09-02 — PACKED pilot on v1.2)

- Batching: **token-budget packing is the default** (A/B gate passed —
  runs/m3_1/ab_packing.json): 40k pair tokens per optimizer step, 5k-token
  length-bucketed micros (4 buckets, rotated every micro, ≤48 rows), in-micro
  contrastive negatives capped at 31 per anchor.
- Corpus: pilot_v1.2 — 375,671 pairs, 44.8M src+tgt tokens.
- Steps per pass: **1,120**. Seconds per step: **~31.5** cold, expect ~33–36
  heat-soaked. Throughput: **~1,350 pair-tokens/s** (unpacked was ~740).
- **One full pass ≈ 10–11 h** (was ~19 h unpacked).
- A 30-minute window ≈ ~55 steps. An 8-hour overnight ≈ ~850 steps ≈ 76% of
  a pass.
- Warmup is 100 steps (= 4M tokens, matching the old 500 × 7.7k). If you ever
  disable packing, set warmup_steps back to 500.
- Peak training memory ≈ 6.8 GiB allocated (guard armed at 14).
- Mini reports (~30 s: z spectrum, eff. rank, mini AUC, gates) every 50 steps
  until step 1,000 → `runs/<run>/val/mini/`; full validation report
  auto-interval lands near ~175 steps ≈ every 1.5 h wall.
- Loss scale settles in the low thousands after a few early halvings — that
  settling is normal, a later downward staircase is not.

## 7b. Sanity numbers from the 200-step smoke (untrained baselines)

AUC hard 0.56 (chance-ish), AUC random 0.99, NLI correct/swapped/zeroed all
≈ 0.06 (babble), effective rank ≈ 944/1024. Training should pull AUC-hard and
NLI-correct up and OPEN the correct-vs-swapped/zeroed gap; effective rank
should not collapse toward low double digits.

## 8. Disk and housekeeping

- Checkpoints: **480 MB** each (trainables fp32 + full 8-bit optimiser state);
  the run keeps the last 3 plus milestones → ~1.5 GB steady state per run.
  Val reports are ~0.7 MB of self-contained HTML each.
- `runs/<name>/` holds: checkpoints, `tb/` (TensorBoard), `steps.jsonl`,
  `telemetry.csv`, `val/step_*/report.html`, `config_snapshot.yaml`.
- Safe to delete: old `runs/_resume_*` (resume-proof artifacts), `runs/preflight`.
