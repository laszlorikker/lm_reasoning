"""Per-chunk attention pooling head (PHASE1_PLAN §2.1 encoder step 4; M2 spec §2).

Single-query multi-head attention with a learned query over the chunk's
layer-SPLIT token states, masked to the span, then LN → MLP → LN on z.
The MLP width follows the attention output width (n_heads·head_dim = 1024):
1024 → 4096 → d_z (recorded in M2_DESIGN.md). z carries no position info.
"""

import torch
import torch.nn as nn


class ChunkPooler(nn.Module):
    def __init__(self, hidden: int, d_z: int, n_heads: int = 8, head_dim: int = 128):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        inner = n_heads * head_dim
        self.query = nn.Parameter(torch.randn(n_heads, head_dim) * 0.02)
        self.k = nn.Linear(hidden, inner, bias=False)
        self.v = nn.Linear(hidden, inner, bias=False)
        self.out = nn.Linear(inner, inner, bias=False)
        self.ln = nn.LayerNorm(inner)
        self.mlp = nn.Sequential(
            nn.Linear(inner, 4 * inner), nn.GELU(), nn.Linear(4 * inner, d_z)
        )
        self.z_ln = nn.LayerNorm(d_z)

    def forward(self, states: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
        """states [B,S,h] (layer-SPLIT residual stream); chunk_mask [B,K,S] bool
        (True where token s belongs to chunk k) → z [B,K,d_z].

        Chunks whose mask is empty (padding chunks) yield a defined but
        meaningless vector; z_mask excludes them everywhere downstream.
        """
        B, S, _ = states.shape
        k = self.k(states).view(B, S, self.n_heads, self.head_dim)
        v = self.v(states).view(B, S, self.n_heads, self.head_dim)
        scores = torch.einsum("hd,bshd->bhs", self.query.to(k.dtype), k) / self.head_dim**0.5
        scores = scores.unsqueeze(1).masked_fill(~chunk_mask.unsqueeze(2), float("-inf"))
        attn = torch.nan_to_num(scores.softmax(dim=-1))  # empty chunks: -inf row -> NaN -> 0
        pooled = torch.einsum("bkhs,bshd->bkhd", attn, v).reshape(B, chunk_mask.shape[1], -1)
        return self.z_ln(self.mlp(self.ln(self.out(pooled))))
