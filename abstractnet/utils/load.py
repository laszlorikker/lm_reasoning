"""Base-LM loading shared by the M0 scripts; M2's AbstractLM builds on the same call."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from abstractnet.config import ModelCfg

DTYPES = {"float16": torch.float16, "float32": torch.float32}


def load_base_lm(cfg: ModelCfg, device: str = "cuda"):
    if cfg.dtype not in DTYPES:
        # bf16 deliberately absent: sm_75 has no bf16 support (PHASE1_PLAN §1)
        raise ValueError(f"dtype must be one of {sorted(DTYPES)} on sm_75, got {cfg.dtype!r}")
    if cfg.load_in_4bit:
        raise NotImplementedError("4-bit base weights are the Qwen3-4B upgrade path; wired in M2")
    dtype = DTYPES[cfg.dtype]
    kwargs = dict(attn_implementation=cfg.attn_implementation, low_cpu_mem_usage=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.name_or_path, dtype=dtype, **kwargs)
    except TypeError:  # transformers < 4.56 spells it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(cfg.name_or_path, torch_dtype=dtype, **kwargs)
    if model.dtype != dtype:
        raise RuntimeError(f"model loaded as {model.dtype}, wanted {dtype} — dtype kwarg ignored?")
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.name_or_path)
    return model, tokenizer
