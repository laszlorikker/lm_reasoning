# WORKSTATION.md — bring-up for the second training machine

**This machine is remote: SSH only, no LAN to the laptop, Hugging Face
BLOCKED, GitHub reachable.** All heavy artifacts arrive exclusively via
GitHub Releases on the private repo (`scripts/export_offline_bundle.py` on the
laptop → `scripts/import_offline_bundle.py` here). The laptop stays a working
rig: nothing here may break the sm_75 path.

## 0. Remote-session basics

- `gh auth login` once (device flow works over SSH), scope: repo.
- Run EVERYTHING long-lived inside `tmux` — an SSH drop kills foreground
  processes. `tmux new -s work` / detach `Ctrl-b d` / `tmux attach -t work`.
- Checking a run from a fresh SSH session:
  `tmux ls` → attach; or without attaching:
  `tail -2 runs/<run>/steps.jsonl` (step advancing?), `nvidia-smi`,
  `ps aux | grep abstractnet.train`.
- After cloning: `git config core.hooksPath scripts/git_hooks`
  (the >100 MB pre-commit guard — do this on every clone).

## 1. Clone and environment

```bash
git clone https://github.com/<you>/<repo>.git reasoning_lm && cd reasoning_lm
git config core.hooksPath scripts/git_hooks
conda create -n abstractnet python=3.12 -y
conda activate abstractnet
# torch first, from the CUDA index matching your driver (nvidia-smi shows the
# supported CUDA version; cu126 works on 12.4+ drivers, use cu128+ on newer):
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
python -m spacy download en_core_web_sm
```

PyPI blocked too, or `pip check` failing? Ask for the bundle to be re-exported
with `--with-wheelhouse`, then `pip install --no-index --find-links wheelhouse -r requirements.txt`.
(spaCy's model is a pip package — it rides in the wheelhouse in that case.)

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

## 4. Data + models — the offline bundle (the only path; HF is blocked)

Fixtures (val pool v3, panel, audit samples) arrive via git. Everything heavy
comes from the GitHub Release:

```bash
python scripts/import_offline_bundle.py --version bundle-v1
conda deactivate && conda activate abstractnet   # pick up the offline env hook
```

The import downloads the release assets, verifies every sha256 against the
manifest, unpacks the model snapshots into `HF_HOME` and the v1.3 corpus
shards into `data/processed/`, writes `HF_HOME` / `HF_HUB_OFFLINE=1` /
`HF_DATASETS_OFFLINE=1` into the conda activation hook, and then runs the
gates: offline model field-check, offline v1.3 finalize (rebuilds `full`
through the real dedup gate), and an exact match of the finalize stats against
DATA.md. Expect "IMPORT COMPLETE — all gates passed".

Then:
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
