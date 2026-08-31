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


@dataclass
class DataCfg:
    max_source_tokens: int = 512
    max_target_tokens: int = 512
    max_chunks: int = 8
    max_chunk_tokens: int = 64


@dataclass
class TrainCfg:
    micro_batch_size: int = 8


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
