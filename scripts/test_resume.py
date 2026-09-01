#!/usr/bin/env python
"""M3 resume proof (spec item 4) — run BEFORE any long run.

A: 300 steps uninterrupted, fixed seed, batch IDs + losses logged per step.
B: same seed, SIGKILLed at a random step in [100, 200] (crash, not clean exit),
   then --resume auto to 300.
C: same, but SIGINT (clean stop path) instead of the kill.

Asserts (exit non-zero on any failure):
  - post-resume batch-ID sequence identical to A, no repeats or gaps in the
    effective trajectory (replayed steps after a crash overwrite their originals);
  - per-step losses match A within fp16 tolerance (actual max deviation printed);
  - LR matches exactly at every step; final optimiser step count matches.

Usage: python scripts/test_resume.py [--config configs/resume_test.yaml] [--steps 300]
"""

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable


def read_steps(run_dir: Path) -> dict[int, dict]:
    """steps.jsonl → {step: record}, keeping the LAST occurrence per step:
    after a crash, replayed steps supersede their pre-crash originals."""
    out: dict[int, dict] = {}
    with open(run_dir / "steps.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["step"]] = rec
    return out


def launch(cfg: str, run: str, steps: int, resume: bool = False) -> subprocess.Popen:
    cmd = [PY, "-m", "abstractnet.train", "--config", cfg, "--run", run, "--steps", str(steps)]
    if resume:
        cmd += ["--resume", "auto"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_for_step(run_dir: Path, target: int, proc: subprocess.Popen, timeout_s: float = 3600):
    t0 = time.monotonic()
    path = run_dir / "steps.jsonl"
    while time.monotonic() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"process exited early (rc={proc.returncode})")
        if path.exists():
            last = 0
            with open(path) as f:
                for line in f:
                    last = json.loads(line)["step"]
            if last >= target:
                return
        time.sleep(2)
    raise TimeoutError(f"never reached step {target}")


def interrupted_run(cfg: str, run: str, steps: int, kill_step: int, sig: int) -> None:
    proc = launch(cfg, run, steps)
    try:
        wait_for_step(Path(run), kill_step, proc)
    finally:
        if proc.poll() is None:
            proc.send_signal(sig)
            if sig == signal.SIGKILL:
                proc.wait(timeout=30)
            else:
                out, _ = proc.communicate(timeout=600)
                assert proc.returncode == 0, f"clean stop must exit 0, got {proc.returncode}\n{out[-2000:]}"
    resumed = launch(cfg, run, steps, resume=True)
    out, _ = resumed.communicate(timeout=3600)
    assert resumed.returncode == 0, f"resumed run failed\n{out[-3000:]}"


def compare(a: dict[int, dict], b: dict[int, dict], steps: int, label: str) -> dict:
    assert sorted(b) == list(range(1, steps + 1)), \
        f"{label}: effective trajectory has gaps/extras: {len(b)} steps"
    max_rel = 0.0
    worst = None
    for s in range(1, steps + 1):
        ra, rb = a[s], b[s]
        assert ra["batch_ids"] == rb["batch_ids"], f"{label}: batch IDs differ at step {s}"
        assert ra["lr"] == rb["lr"], f"{label}: LR differs at step {s}: {ra['lr']} vs {rb['lr']}"
        denom = max(abs(ra["loss"]), 1e-8)
        rel = abs(ra["loss"] - rb["loss"]) / denom
        if rel > max_rel:
            max_rel, worst = rel, s
    assert a[steps]["opt_steps"] == b[steps]["opt_steps"], \
        f"{label}: final optimiser step count differs: " \
        f"{a[steps]['opt_steps']} vs {b[steps]['opt_steps']}"
    tol = 1e-2  # fp16 accumulation tolerance; actual deviation reported below
    assert max_rel <= tol, f"{label}: loss deviates rel {max_rel:.2e} at step {worst} (tol {tol})"
    return {"max_rel_loss_dev": max_rel, "worst_step": worst,
            "opt_steps": b[steps]["opt_steps"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/resume_test.yaml")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    random.seed(os.getpid())

    runs = {name: Path(f"runs/_resume_{name}") for name in "ABC"}
    for r in runs.values():
        subprocess.run(["rm", "-rf", str(r)], check=True)

    print("[A] uninterrupted reference run…")
    pa = launch(args.config, str(runs["A"]), args.steps)
    out, _ = pa.communicate(timeout=7200)
    assert pa.returncode == 0, f"run A failed\n{out[-3000:]}"
    a = read_steps(runs["A"])

    kill_step = random.randint(100, 200)
    print(f"[B] crash run: SIGKILL at step ~{kill_step}, then --resume auto…")
    interrupted_run(args.config, str(runs["B"]), args.steps, kill_step, signal.SIGKILL)
    rb = compare(a, read_steps(runs["B"]), args.steps, "B(SIGKILL)")
    ckpts_b = sorted(runs["B"].glob("step_*.pt"))
    print(f"[B] OK — max rel loss dev {rb['max_rel_loss_dev']:.2e} at step {rb['worst_step']}, "
          f"opt_steps {rb['opt_steps']}, checkpoints kept {len(ckpts_b)}")

    int_step = random.randint(100, 200)
    print(f"[C] clean-stop run: SIGINT at step ~{int_step}, then --resume auto…")
    interrupted_run(args.config, str(runs["C"]), args.steps, int_step, signal.SIGINT)
    rc = compare(a, read_steps(runs["C"]), args.steps, "C(SIGINT)")
    print(f"[C] OK — max rel loss dev {rc['max_rel_loss_dev']:.2e} at step {rc['worst_step']}, "
          f"opt_steps {rc['opt_steps']}")

    print(json.dumps({"sigkill": rb, "sigint": rc}, indent=2))
    print("RESUME PROOF PASSED")


if __name__ == "__main__":
    main()
