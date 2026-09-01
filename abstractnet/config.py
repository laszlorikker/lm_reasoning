"""YAML -> typed config. Strict: unknown keys or sections are errors (typo safety).

Every experimental knob lives in configs/*.yaml (PHASE1_PLAN §5, kickoff rule 6);
the Qwen3-4B upgrade must be expressible as a config change only.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class ModelCfg:
    name_or_path: str = "Qwen/Qwen3-1.7B-Base"
    attn_implementation: str = "sdpa"
    dtype: str = "float16"
    load_in_4bit: bool = False
    split_layer: int = 14
    d_z: int = 1024
    # M2: encoder/decoder additions (spec of 2026-09-01)
    languages: tuple = ("en", "fr", "de", "es", "it", "pt")
    k_max: int = 8
    pool_heads: int = 8
    pool_head_dim: int = 128
    xattn_every: int = 2
    xattn_heads: int = 8
    xattn_head_dim: int = 128
    rate: str = "noise"          # noise | kl
    rate_sigma: float = 0.1
    word_dropout: float = 0.2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # named adapter on upper layers of the LOWER half, encoder pass only;
    # empty = not created (M5 ablation hook, off through M2/M3)
    encoder_lora_layers: tuple = ()


@dataclass
class DataCfg:
    max_source_tokens: int = 512
    max_target_tokens: int = 512
    max_chunks: int = 8
    max_chunk_tokens: int = 64
    corpus_path: str = "data/processed/pilot_v1.1/full"
    val_pool_path: str = "data/fixtures/val_pool_v2.jsonl"
    panel_path: str = "data/fixtures/panel_v1.jsonl"
    # pilot-corpus recipe (source budgets, negative shares, K targets);
    # validated by scripts/build_pilot.py, which owns its schema
    pilot: dict | None = None


@dataclass
class TrainCfg:
    micro_batch_size: int = 8
    # M2: loss weights and schedule knobs (PHASE1_PLAN §4)
    tau: float = 0.05
    lambda_c: float = 0.1
    lambda_r: float = 1.0e-3
    chunk_ce_size: int = 1024
    p_id_start: float = 0.2
    p_id_end: float = 0.05
    p_id_decay_frac: float = 0.2   # p_id decays over the first fraction of steps
    # M3: optimiser / schedule / run mechanics (PHASE1_PLAN §5, M3 spec)
    effective_batch: int = 64      # via gradient accumulation
    lr_new: float = 2.0e-4
    lr_lora: float = 1.0e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    epochs: int = 1
    max_steps: int = 0             # 0 = derive from epochs x corpus size
    seed: int = 1234
    log_interval: int = 10
    eval_interval_steps: int = 0   # 0 = auto from measured step + report time (<10% rule)
    checkpoint_interval_steps: int = 500
    checkpoint_interval_minutes: float = 15.0
    keep_checkpoints: int = 3
    telemetry_interval_s: float = 30.0


@dataclass
class ProfileCfg:
    batch_size: int = 8
    seq_len: int = 512
    vram_budget_gib: float = 14.0


@dataclass
class Config:
    model: ModelCfg
    data: DataCfg
    train: TrainCfg
    profile: ProfileCfg


_SECTIONS = {"model": ModelCfg, "data": DataCfg, "train": TrainCfg, "profile": ProfileCfg}


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise KeyError(f"unknown config sections {sorted(unknown)} in {path}; known: {sorted(_SECTIONS)}")
    built = {}
    for name, cls in _SECTIONS.items():
        section = raw.get(name) or {}
        known = {f.name for f in fields(cls)}
        bad = set(section) - known
        if bad:
            raise KeyError(f"unknown keys {sorted(bad)} in section '{name}' of {path}; known: {sorted(known)}")
        built[name] = cls(**section)
    return Config(**built)
