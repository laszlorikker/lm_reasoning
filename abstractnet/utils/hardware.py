"""Hardware detection (workstation migration): precision and attention backend
derive from compute capability; the memory budget stays in the per-machine
config (profile.vram_budget_gib), never in code.

sm_80+ : bf16 autocast (no GradScaler), SDPA auto-selects the flash backend.
< sm_80: fp16 autocast + GradScaler, SDPA mem-efficient backend (the gated
         laptop path — must keep passing unchanged).
"""

import torch


def detect() -> dict:
    cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    bf16 = cap >= (8, 0)
    return {
        "name": torch.cuda.get_device_name(0),
        "capability": list(cap),
        "total_gib": round(props.total_memory / 1024**3, 1),
        "bf16_supported": bf16,
        "flash_sdpa_supported": bf16,  # PyTorch's flash SDPA kernel needs sm_80+
        "recommended_dtype": "bfloat16" if bf16 else "float16",
    }
