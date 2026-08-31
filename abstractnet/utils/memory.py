"""CUDA memory bookkeeping, used by every milestone's memory profile (kickoff rule 4)."""

import gc
import subprocess

import torch

GIB = 1024**3


def reset_peak() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_gib() -> tuple[float, float]:
    """(max allocated, max reserved) since the last reset, in GiB."""
    return (
        torch.cuda.max_memory_allocated() / GIB,
        torch.cuda.max_memory_reserved() / GIB,
    )


def resident_gib() -> float:
    return torch.cuda.memory_allocated() / GIB


def driver_used_total_gib() -> tuple[float, float]:
    """Device-wide (used, total) as the driver sees it — includes the CUDA
    context, allocator cache, and anything Windows holds on this WSL2 box."""
    free, total = torch.cuda.mem_get_info()
    return (total - free) / GIB, total / GIB


def nvidia_smi_sample() -> str:
    """One 'sm_clock_mhz, power_w, temp_c, mem_used_mib' CSV row."""
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=clocks.sm,power.draw,temperature.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
