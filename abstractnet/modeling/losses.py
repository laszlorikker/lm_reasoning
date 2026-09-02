"""Reusable losses (M2 spec §5).

chunked_lm_loss — full-vocab cross-entropy over flattened positions,
CHUNK_SIZE at a time, each chunk's logits recomputed during backward
(torch.utils.checkpoint), per the M0 memory findings. Tested against the naive
loss for equality of value AND gradient in fp32.

symmetric_info_nce — InfoNCE (τ) on L2-normalised document vectors, in-batch
negatives plus explicit extra negatives, symmetric in the two views.
"""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint


def chunked_lm_loss(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 1024,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """hidden [B,T,h] — states whose position t predicts labels[:, t];
    labels [B,T] with -100 = ignore. Mean CE over non-ignored positions."""
    h = hidden.flatten(0, 1)
    t = labels.flatten()
    n = (t != -100).sum().clamp(min=1)

    def chunk_loss(hc: torch.Tensor, tc: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hc, lm_head_weight)
        return F.cross_entropy(logits.float(), tc, ignore_index=-100, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], chunk_size):
        hc, tc = h[i : i + chunk_size], t[i : i + chunk_size]
        if use_checkpoint and hidden.requires_grad:
            total = total + torch.utils.checkpoint.checkpoint(chunk_loss, hc, tc, use_reentrant=False)
        else:
            total = total + chunk_loss(hc, tc)
    return total / n


def doc_vectors(z_clean: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over chunks → L2-normalised document vector [*, d_z], fp32."""
    z = z_clean.float()
    m = z_mask.unsqueeze(-1).float()
    v = (z * m).sum(-2) / m.sum(-2).clamp(min=1e-6)
    return F.normalize(v, dim=-1)


def symmetric_info_nce(
    va: torch.Tensor, vb: torch.Tensor, v_neg: torch.Tensor | None, tau: float = 0.05,
    max_inbatch_negs: int | None = None, seed: int = 0,
) -> torch.Tensor:
    """va, vb [B,d] normalised views (positives pair row-wise); v_neg [N,d]
    explicit hard-negative columns for both directions.

    max_inbatch_negs (M3.1 guard c): with packed variable-size micros, the
    in-batch negative count per anchor is capped by seeded subsampling to keep
    the InfoNCE denominator in a narrow band across micros; the positive and
    ALL explicit hard-negative columns are always kept. Deterministic in seed.
    """
    va, vb = va.float(), vb.float()
    neg = v_neg.float() if v_neg is not None and v_neg.numel() else None
    B = va.shape[0]
    targets = torch.arange(B, device=va.device)

    def keep_mask(direction_seed: int) -> torch.Tensor | None:
        if max_inbatch_negs is None or B - 1 <= max_inbatch_negs:
            return None
        g = torch.Generator().manual_seed(direction_seed)
        scores = torch.rand(B, B, generator=g)
        scores.fill_diagonal_(2.0)  # the positive column is always kept
        thresh = scores.topk(max_inbatch_negs + 1, dim=1).values[:, -1:]
        return (scores >= thresh).to(va.device)

    def one_way(x, y, direction_seed):
        logits = x @ y.t()
        km = keep_mask(direction_seed)
        if km is not None:
            logits = logits.masked_fill(~km, float("-inf"))
        if neg is not None:
            logits = torch.cat([logits, x @ neg.t()], dim=1)
        return F.cross_entropy(logits / tau, targets)

    return 0.5 * (one_way(va, vb, seed) + one_way(vb, va, seed + 1))
