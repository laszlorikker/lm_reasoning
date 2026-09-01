"""Rate term on z (PHASE1_PLAN §2.1 encoder step 5; M2 spec §2).

`noise`: absolute Gaussian noise (σ) after the final LayerNorm, training only.
`kl`:    VIB — μ and logσ² heads, reparameterised sample in training, KL to
         N(0, I) averaged over real (z_mask) chunks.

Contract: forward(z, z_mask, training) → (z_clean, z_out, kl).
z_clean feeds the contrastive term and every eval; z_out feeds the decoder.
"""

import torch
import torch.nn as nn


class NoiseRate(nn.Module):
    def __init__(self, d_z: int, sigma: float = 0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, z, z_mask, training: bool):
        z_out = z + self.sigma * torch.randn_like(z) if training else z
        return z, z_out, z.new_zeros(())


class KLRate(nn.Module):
    def __init__(self, d_z: int, sigma: float = 0.0):
        super().__init__()
        self.mu = nn.Linear(d_z, d_z)
        self.logvar = nn.Linear(d_z, d_z)

    def forward(self, z, z_mask, training: bool):
        mu, logvar = self.mu(z), self.logvar(z).clamp(-8, 8)
        if training:
            z_out = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            z_out = mu
        kl_per_chunk = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1)  # [B,K]
        denom = z_mask.sum().clamp(min=1)
        kl = (kl_per_chunk * z_mask).sum() / denom
        return mu, z_out, kl


def make_rate(mode: str, d_z: int, sigma: float) -> nn.Module:
    if mode == "noise":
        return NoiseRate(d_z, sigma)
    if mode == "kl":
        return KLRate(d_z)
    raise ValueError(f"rate mode must be 'noise' or 'kl', got {mode!r}")
