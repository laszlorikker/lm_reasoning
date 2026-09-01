"""Gated cross-attention wrapper (PHASE1_PLAN §2.1 decoder step 2; M2 spec §3).

Wraps an original decoder layer: LN → cross-attention (queries from the hidden
state; keys/values from z already projected to hidden size, chunk-position
embedding added on the key/value side by AbstractLM) → tanh(gate)·out added to
the residual → the wrapped layer. The gate is a scalar per block, initialised
to ZERO, so at step 0 the decoder is exactly the base LM. No gated FFN in v1.

z context is set per forward via set_z()/clear_z() (AbstractLM manages it);
with no z set, the wrapper is a transparent pass-through — that is the
base-equality configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedXAttnLayer(nn.Module):
    def __init__(self, layer: nn.Module, hidden: int, n_heads: int = 8, head_dim: int = 128):
        super().__init__()
        self.layer = layer
        self.n_heads, self.head_dim = n_heads, head_dim
        inner = n_heads * head_dim
        self.ln = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, inner, bias=False)
        self.k = nn.Linear(hidden, inner, bias=False)
        self.v = nn.Linear(hidden, inner, bias=False)
        self.o = nn.Linear(inner, hidden, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))
        self._z_ctx: tuple[torch.Tensor, torch.Tensor] | None = None

    def set_z(self, z_h: torch.Tensor, z_mask: torch.Tensor) -> None:
        """z_h [B,K,hidden] (projected z + chunk-position embedding), z_mask [B,K].
        Every sample must have z_mask.any() — an all-masked row would NaN."""
        self._z_ctx = (z_h, z_mask)

    def clear_z(self) -> None:
        self._z_ctx = None

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        if self._z_ctx is not None:
            z_h, z_mask = self._z_ctx
            B, T, _ = hidden_states.shape
            K = z_h.shape[1]
            x = self.ln(hidden_states)
            q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k(z_h).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v(z_h).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
            attn_mask = z_mask[:, None, None, :]  # [B,1,1,K], True = attend
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            out = self.o(out.transpose(1, 2).reshape(B, T, -1))
            hidden_states = hidden_states + torch.tanh(self.gate) * out
        return self.layer(hidden_states, *args, **kwargs)
