# WORKSTATION.md — bring-up for the second training machine

Order matters. The laptop stays a working rig: nothing here may break the
sm_75 path (its gates are re-run in CI-fashion after every hardware-aware
change).

## 1. Clone and environment

```bash
git clone git@github.com:<you>/<repo>.git reasoning_lm && cd reasoning_lm
conda create -n abstractnet python=3.12 -y
conda activate abstractnet
# torch first, from the CUDA index matching your driver (nvidia-smi shows the
# supported CUDA version; cu126 works on 12.4+ drivers, use cu128+ on newer):
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
python -m spacy download en_core_web_sm
```

## 2. Hardware gates

```bash
python scripts/check_env.py --with-bnb
```

The gates are hardware-aware: on sm_80+ they verify bf16 and the flash SDPA
backend (no GradScaler); below sm_80 the original fp16 + mem-efficient +
GradScaler gates run unchanged. All must PASS before anything else.

## 3. Per-machine config

```bash
cp configs/base.yaml configs/workstation.yaml
```

Edit `configs/workstation.yaml`:
- `model.dtype: bfloat16` (sm_80+ only — the loader refuses it below that)
- `profile.vram_budget_gib:` GPU total minus ~2 GiB (e.g. 22.0 on a 24 GiB card)
- `data.corpus_path: data/processed/pilot_v1.3/full` (pilot-2 trains on v1.3)
- leave `train.*` as committed (packing gated, warmup 100); revisit
  `micro_token_budget`/`token_budget` only after a batch sweep on this GPU
  (`python scripts/batch_sweep.py`) — more VRAM likely supports larger micros.

Use `--config configs/workstation.yaml` in every command below.

## 4. Data

Fixtures (val pool v3, panel, audit samples) arrive via git. The corpus does
not. Two options:

**(a) Rebuild (idempotent, needs network):**
```bash
python scripts/verify_datasets.py            # streaming probes, no bulk downloads
python scripts/build_pilot.py --config configs/workstation.yaml --out data/processed/pilot_v1.3
```
Downloads several GB of datasets — including the **~6 GB OpenMathInstruct-2
train_5M** pull again. Deterministic: same seeds → same corpus; the finalize
re-runs the dedup gate (the eval-hash cache rebuilds itself).

**(b) rsync from the laptop over LAN (minutes, byte-identical):**
```bash
rsync -avP laszlo@<laptop-ip>:~/reasoning_lm/data/processed/pilot_v1.3 data/processed/
rsync -avP laszlo@<laptop-ip>:~/reasoning_lm/runs/m1/eval_hashes.txt runs/m1/
```

Then verify either way:
```bash
python -m abstractnet.train --config configs/workstation.yaml --run runs/preflight --dry-run
```

## 5. Smoke before anything long

```bash
python -m abstractnet.train --config configs/workstation.yaml --run runs/ws_smoke --steps 200
```

Watch the mini reports (`runs/ws_smoke/val/mini/`, every 50 steps) and the
step log. Optional but recommended once per machine:
`python scripts/test_resume.py --config configs/resume_test_packed.yaml --steps 150`
(copy the resume-test config and set its corpus/dtype for this machine first).
Note the resume guarantee is per-machine: cross-machine `--resume` loads
everything and warns, but bit-exactness is only proven same-machine.

## 6. Pilot-2

```bash
tmux new -s pilot2
python -m abstractnet.train --config configs/workstation.yaml --run runs/pilot2 --max-wall-minutes 0
tensorboard --logdir runs/pilot2/tb --port 6006
```

RUNBOOK.md §4 (what healthy looks like, red flags) applies unchanged; its §7
throughput numbers are laptop numbers — expect this machine to set its own via
the smoke and sweep.
