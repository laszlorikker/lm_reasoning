"""Phase-1 training loop (M3). Config-driven; the OPERATOR starts, monitors and
stops runs (see RUNBOOK.md) — this module's job is determinism, full-state
checkpointing, and honest logging.

    python -m abstractnet.train --config configs/base.yaml --run runs/<name>
    python -m abstractnet.train ... --resume auto
    python -m abstractnet.train ... --dry-run

Checkpoints carry FULL training state: trainable weights, AdamW8bit state,
GradScaler, LR scheduler, step/epoch counters, data-sampler position, RNG
states (python / numpy / torch CPU+CUDA), the config snapshot, and the git
commit. Saves are atomic (tmp + rename). SIGINT/SIGTERM, a STOP sentinel file,
and --max-wall-minutes all finish the current step, save, and exit 0.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from abstractnet.config import Config, load_config
from abstractnet.data.collate import PairCollator
from abstractnet.utils import memory as mem

CKPT_GLOB = "step_*.pt"


# --------------------------------------------------------------- scheduling


def p_id_at(step: int, total_steps: int, tc) -> float:
    decay_steps = max(1, int(total_steps * tc.p_id_decay_frac))
    frac = min(1.0, step / decay_steps)
    return tc.p_id_start + frac * (tc.p_id_end - tc.p_id_start)


def lr_lambda(step: int, total_steps: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    import math

    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ------------------------------------------------------------------ sampler


class EpochSampler:
    """Deterministic epoch-shuffled sequential sampler with an exact position
    state (epoch, index) — the unit of resume."""

    def __init__(self, ds, micro_bs: int, seed: int):
        self.ds, self.micro_bs, self.seed = ds, micro_bs, seed
        self.epoch, self.index = 0, 0
        self._perm = None

    def state(self) -> dict:
        return {"mode": "epoch", "epoch": self.epoch, "index": self.index, "seed": self.seed}

    def load_state(self, s: dict) -> None:
        assert s.get("mode", "epoch") == "epoch", "checkpoint was written by a packed sampler"
        assert s["seed"] == self.seed, "sampler seed differs from checkpoint"
        self.epoch, self.index = s["epoch"], s["index"]
        self._perm = None

    def steps_per_epoch(self, accum: int) -> int:
        return len(self.ds) // (self.micro_bs * accum)

    def next_rows(self) -> list[dict]:
        if self._perm is None:
            self._perm = self.ds.shuffle(seed=self.seed + 1000 * self.epoch)
        if self.index + self.micro_bs > len(self._perm):
            self.epoch += 1
            self.index = 0
            self._perm = self.ds.shuffle(seed=self.seed + 1000 * self.epoch)
        rows = [dict(self._perm[i]) for i in range(self.index, self.index + self.micro_bs)]
        self.index += self.micro_bs
        return rows


# -------------------------------------------------------------- checkpoints


def rng_states() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }


def restore_rng(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch_cpu"].cpu() if torch.is_tensor(s["torch_cpu"]) else s["torch_cpu"])
    torch.cuda.set_rng_state(s["torch_cuda"].cpu() if torch.is_tensor(s["torch_cuda"]) else s["torch_cuda"])


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def save_checkpoint(run_dir: Path, model, opt, scaler, step: int, opt_steps: int,
                    sampler: EpochSampler, cfg_text: str, keep: int) -> Path:
    from abstractnet.utils.hardware import detect

    payload = {
        "trainable": model.trainable_state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "opt_steps": opt_steps,
        "sampler": sampler.state(),
        "rng": rng_states(),
        "config_text": cfg_text,
        "git": git_hash(),
        "hardware": detect(),
        "saved_at": time.time(),
    }
    path = run_dir / f"step_{step:06d}.pt"
    tmp = run_dir / f".tmp_{path.name}"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on the same filesystem
    ckpts = sorted(run_dir.glob(CKPT_GLOB))
    for old in ckpts[:-keep]:
        old.unlink()
    return path


def find_latest_valid(run_dir: Path):
    """Newest checkpoint that actually loads (a kill during save leaves only
    the atomic previous ones, but be defensive anyway)."""
    for path in sorted(run_dir.glob(CKPT_GLOB), reverse=True):
        try:
            return path, torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[resume] skipping unreadable {path.name}: {e}")
    return None, None


# ---------------------------------------------------------------- telemetry


def telemetry_thread(stop: threading.Event, path: Path, interval_s: float) -> None:
    with open(path, "a") as f:
        if f.tell() == 0:
            f.write("t_unix,sm_clock_mhz,power_w,temp_c,mem_used_mib\n")
        while not stop.is_set():
            try:
                f.write(f"{time.time():.0f},{mem.nvidia_smi_sample()}\n")
                f.flush()
            except Exception:
                pass
            stop.wait(interval_s)


# ------------------------------------------------------------------ dry run


def dry_run(cfg: Config, tokenizer) -> None:
    from datasets import load_from_disk

    ds = load_from_disk(cfg.data.corpus_path)
    need = {"id", "src_ids", "src_spans", "tgt_ids", "tgt_spans", "lang_src",
            "lang_tgt", "neg_ids", "neg_spans", "n_src_tokens", "n_tgt_tokens", "k"}
    missing = need - set(ds.column_names)
    assert not missing, f"corpus schema missing columns: {missing}"
    src_tok = sum(ds["n_src_tokens"])
    tgt_tok = sum(ds["n_tgt_tokens"])
    tc = cfg.train
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = PairCollator(pad_id=pad)
    print(f"corpus: {cfg.data.corpus_path}")
    print(f"rows: {len(ds):,}   src+tgt tokens: {src_tok + tgt_tok:,}")
    if tc.packing_enabled:
        from abstractnet.data.packing import BucketedSampler

        s = BucketedSampler(ds, tc.seed, tc.n_buckets, tc.micro_token_budget,
                            tc.max_rows_per_micro, tc.token_budget)
        plan = s.plan_step()
        for rows in plan[:3]:
            collator(rows)
        sizes = [len(m) for m in plan]
        toks = [sum(len(r["src_ids"]) + len(r["tgt_ids"]) for r in m) for m in plan]
        print(f"packed: token_budget {tc.token_budget}, micro budget {tc.micro_token_budget}, "
              f"{tc.n_buckets} buckets")
        print(f"first step plan: {len(plan)} micros, rows/micro {sizes}, tokens/micro {toks}")
        print(f"bucket sizes (rows): "
              f"{ {b: len(v) for b, v in s.rows_by_bucket.items()} }")
        print(f"steps per pass: {s.steps_per_epoch():,}   "
              f"(epochs={tc.epochs} -> total {s.steps_per_epoch() * tc.epochs:,})")
    else:
        accum = tc.effective_batch // tc.micro_batch_size
        steps = len(ds) // tc.effective_batch
        sampler = EpochSampler(ds, tc.micro_batch_size, tc.seed)
        for _ in range(3):
            collator(sampler.next_rows())
        print(f"micro {tc.micro_batch_size} x accum {accum} = effective {tc.effective_batch}")
        print(f"steps per epoch: {steps:,}   (epochs={tc.epochs} -> total {steps * tc.epochs:,})")
    print("collation of sample micro-batches: OK")
    print("DRY RUN OK")


# --------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--run", required=True, help="run directory, e.g. runs/pilot1")
    ap.add_argument("--resume", default=None, help="'auto' or a checkpoint path")
    ap.add_argument("--allow-config-change", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-wall-minutes", type=float, default=0.0,
                    help="save and exit after this many minutes (0 = unlimited)")
    ap.add_argument("--steps", type=int, default=0, help="override total steps (smoke runs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg_text = Path(args.config).read_text()
    tc = cfg.train
    run_dir = Path(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text(cfg_text)

    from transformers import AutoTokenizer

    if args.dry_run:
        dry_run(cfg, AutoTokenizer.from_pretrained(cfg.model.name_or_path))
        return

    import bitsandbytes as bnb
    from datasets import load_from_disk
    from torch.utils.tensorboard import SummaryWriter

    from abstractnet.modeling.abstract_lm import AbstractLM

    mem.apply_budget_guard(cfg.profile.vram_budget_gib)
    torch.manual_seed(tc.seed)
    random.seed(tc.seed)
    np.random.seed(tc.seed)

    model = AbstractLM(cfg)
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    ds = load_from_disk(cfg.data.corpus_path)
    if tc.packing_enabled:
        from abstractnet.data.packing import BucketedSampler

        sampler = BucketedSampler(ds, tc.seed, tc.n_buckets, tc.micro_token_budget,
                                  tc.max_rows_per_micro, tc.token_budget)
        steps_per_epoch = sampler.steps_per_epoch()
        mode = f"packed({tc.token_budget} tok/step)"
    else:
        accum = tc.effective_batch // tc.micro_batch_size
        assert tc.effective_batch % tc.micro_batch_size == 0
        sampler = EpochSampler(ds, tc.micro_batch_size, tc.seed)
        steps_per_epoch = sampler.steps_per_epoch(accum)
        mode = f"unpacked(accum {accum})"
    total_steps = args.steps or tc.max_steps or steps_per_epoch * tc.epochs
    pad = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None \
        else model.tokenizer.eos_token_id
    collator = PairCollator(pad_id=pad)

    new_p = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    lora_p = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    opt = bnb.optim.AdamW8bit(
        [{"params": new_p, "lr": tc.lr_new}, {"params": lora_p, "lr": tc.lr_lora}],
        weight_decay=tc.weight_decay)
    # bf16 needs no loss scaling; a disabled GradScaler is a clean passthrough
    use_scaler = cfg.model.dtype == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    step, opt_steps = 0, 0
    if args.resume:
        if args.resume == "auto":
            path, payload = find_latest_valid(run_dir)
            assert payload is not None, f"--resume auto found no valid checkpoint in {run_dir}"
        else:
            path = Path(args.resume)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        if yaml.safe_load(payload["config_text"]) != yaml.safe_load(cfg_text) \
                and not args.allow_config_change:
            sys.exit("resume refused: config differs from the checkpoint snapshot "
                     "(pass --allow-config-change to override)")
        missing = model._trainable_keys() - set(payload["trainable"])
        assert not missing, f"checkpoint missing trainable keys: {sorted(missing)[:5]}"
        model.load_state_dict(payload["trainable"], strict=False)
        opt.load_state_dict(payload["optimizer"])
        sampler.load_state(payload["sampler"])
        step, opt_steps = payload["step"], payload["opt_steps"]
        # cross-machine tolerance: load everything loadable; bit-exact
        # continuation is guaranteed ONLY on unchanged hardware (test_resume)
        from abstractnet.utils.hardware import detect

        hw_then, hw_now = payload.get("hardware"), detect()
        if hw_then and (hw_then["name"] != hw_now["name"]
                        or hw_then["capability"] != hw_now["capability"]):
            print(f"[resume] WARNING: hardware changed "
                  f"({hw_then['name']} sm_{hw_then['capability']} -> "
                  f"{hw_now['name']} sm_{hw_now['capability']}); all state loaded, "
                  "but bit-exact continuation no longer applies "
                  "(the same-machine resume proof is the only tested guarantee)")
        try:
            scaler.load_state_dict(payload["scaler"])
        except Exception as e:
            print(f"[resume] WARNING: GradScaler state not restored ({e}); "
                  "fresh scaler (expected when precision mode changed)")
        try:
            restore_rng(payload["rng"])
        except Exception as e:
            print(f"[resume] WARNING: RNG state not fully restored ({e}); "
                  "continuation is not bit-exact")
        print(f"[resume] {path.name}: step {step}, sampler {sampler.state()}, "
              f"scale {scaler.get_scale():.0f}, git-then {payload['git'][:8]}")

    writer = SummaryWriter(str(run_dir / "tb"))
    steps_log = open(run_dir / "steps.jsonl", "a")
    stop_event = threading.Event()
    tele = threading.Thread(target=telemetry_thread,
                            args=(stop_event, run_dir / "telemetry.csv", tc.telemetry_interval_s),
                            daemon=True)
    tele.start()

    stop_reason: dict = {"why": None}

    def on_signal(signum, frame):
        stop_reason["why"] = f"signal {signal.Signals(signum).name}"

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    sentinel = run_dir / "STOP"
    t_start = time.monotonic()
    last_save = time.monotonic()
    eval_interval = tc.eval_interval_steps  # 0 = decide after first eval
    next_eval = 200 if eval_interval == 0 else eval_interval
    step_times: list[float] = []
    prev_scale = scaler.get_scale()
    overflow_steps = 0

    print(f"[train] total steps {total_steps}, {mode}, run {run_dir}, "
          f"resume at step {step}")
    mem.reset_peak()
    while step < total_steps:
        t0 = time.monotonic()
        batch_ids: list[str] = []
        loss_acc = recon = con = kl = 0.0
        p_id = p_id_at(step, total_steps, tc)
        collator.p_id = p_id
        if tc.packing_enabled:
            micro_rows = sampler.plan_step()
        else:
            micro_rows = [sampler.next_rows() for _ in range(accum)]
        micro_toks = [sum(len(r["src_ids"]) + len(r["tgt_ids"]) for r in rows)
                      for rows in micro_rows]
        tokens = sum(micro_toks)
        micro_pairs = [len(rows) for rows in micro_rows]
        origins = Counter(r["origin"].split("-")[0] for rows in micro_rows for r in rows)
        for micro, rows in enumerate(micro_rows):
            w = micro_toks[micro] / tokens  # token-weighted micro combination
            collator.reseed(hash((tc.seed, step, micro)) & 0x7FFFFFFF)
            batch = collator(rows)
            batch_ids += batch["meta"]["id"]
            out = model(batch)
            scaler.scale(out["loss"] * w).backward()
            loss_acc += float(out["loss"].detach()) * w
            recon += float(out["recon"]) * w
            con += float(out["contrastive"]) * w
            kl += float(out["rate_kl"]) * w
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(new_p + lora_p, tc.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        new_scale = scaler.get_scale()
        if new_scale < prev_scale:
            overflow_steps += 1
        else:
            opt_steps += 1
        prev_scale = new_scale
        for g in opt.param_groups:
            g["lr"] = (tc.lr_new if g is opt.param_groups[0] else tc.lr_lora) \
                * lr_lambda(step, total_steps, tc.warmup_steps)
        step += 1
        dt = time.monotonic() - t0
        step_times.append(dt)

        epoch_now = sampler.epoch if isinstance(sampler.epoch, int) else min(sampler.epoch.values())
        rec = {"step": step, "epoch": epoch_now, "loss": round(loss_acc, 5),
               "micro_pairs": micro_pairs, "origins": dict(origins),
               "recon": round(recon, 5), "con": round(con, 5), "kl": round(kl, 6),
               "lr": opt.param_groups[0]["lr"], "scale": new_scale, "p_id": round(p_id, 4),
               "dt_s": round(dt, 3), "tok": tokens, "tok_s": round(tokens / dt),
               "overflows": overflow_steps, "opt_steps": opt_steps,
               "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / mem.GIB, 2),
               "batch_ids": batch_ids}
        steps_log.write(json.dumps(rec) + "\n")
        steps_log.flush()
        if step % tc.log_interval == 0 or step == 1:
            writer.add_scalar("loss/total", loss_acc, step)
            writer.add_scalar("loss/recon", recon, step)
            writer.add_scalar("loss/contrastive", con, step)
            writer.add_scalar("loss/rate_kl", kl, step)
            writer.add_scalar("opt/lr_new", opt.param_groups[0]["lr"], step)
            writer.add_scalar("opt/loss_scale", new_scale, step)
            writer.add_scalar("opt/p_id", p_id, step)
            writer.add_scalar("opt/overflow_steps", overflow_steps, step)
            writer.add_scalar("perf/step_s", dt, step)
            writer.add_scalar("perf/pair_tokens_s", tokens / dt, step)
            writer.add_scalar("perf/peak_alloc_gib", rec["peak_alloc_gib"], step)
            for i, g in zip(model.xattn_indices, out["gates"].tolist()):
                writer.add_scalar(f"gates/layer_{i}", g, step)
            writer.add_scalar("z/doc_norm", float(out["z_norm"]), step)
            writer.add_scalar("z/dim_var_mean", float(out["z_dim_var_mean"]), step)
            print(f"step {step:>6}/{total_steps}  loss {loss_acc:.4f}  recon {recon:.4f}  "
                  f"con {con:.4f}  scale {new_scale:.0f}  {dt:.2f}s", flush=True)

        # ---- mini report (first mini_report_until steps only)
        if step <= tc.mini_report_until and step % tc.mini_report_every == 0:
            try:
                from eval import val_report

                val_report.mini_generate(model, cfg, run_dir, step, writer)
            except Exception as e:
                print(f"[mini] failed (training continues): {type(e).__name__}: {e}")
            model.train()

        # ---- validation report
        if step >= next_eval:
            try:
                from eval import val_report

                t_eval = time.monotonic()
                val_report.generate(model=model, cfg=cfg, run_dir=run_dir, step=step,
                                    writer=writer)
                eval_s = time.monotonic() - t_eval
                med = sorted(step_times)[len(step_times) // 2]
                if eval_interval == 0:
                    tc_interval = max(100, int(round(eval_s / (0.10 * med) / 50) * 50))
                    eval_interval = tc_interval
                    print(f"[eval] report took {eval_s:.0f}s; step {med:.2f}s -> "
                          f"interval set to {eval_interval} steps (<10% rule)")
                next_eval = step + eval_interval
                model.train()
            except Exception as e:
                print(f"[eval] report FAILED (training continues): {type(e).__name__}: {e}")
                next_eval = step + (eval_interval or 1000)
                model.train()

        # ---- checkpoint cadence
        if step % tc.checkpoint_interval_steps == 0 or \
                (time.monotonic() - last_save) / 60 >= tc.checkpoint_interval_minutes:
            p = save_checkpoint(run_dir, model, opt, scaler, step, opt_steps,
                                sampler, cfg_text, tc.keep_checkpoints)
            last_save = time.monotonic()
            print(f"[ckpt] {p.name}")

        # ---- stop conditions (after a completed step)
        if sentinel.exists():
            stop_reason["why"] = "STOP sentinel"
        if args.max_wall_minutes and (time.monotonic() - t_start) / 60 >= args.max_wall_minutes:
            stop_reason["why"] = f"wall clock {args.max_wall_minutes} min"
        if stop_reason["why"]:
            break

    p = save_checkpoint(run_dir, model, opt, scaler, step, opt_steps,
                        sampler, cfg_text, tc.keep_checkpoints)
    stop_event.set()
    steps_log.close()
    writer.close()
    if sentinel.exists():
        sentinel.unlink()
    why = stop_reason["why"] or "completed"
    print(f"[train] stopped ({why}) at step {step}/{total_steps}; saved {p.name}; "
          f"overflow steps {overflow_steps}")


if __name__ == "__main__":
    main()
