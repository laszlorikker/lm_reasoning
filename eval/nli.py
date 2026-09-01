"""Bidirectional NLI scorer for meaning-preservation (PHASE1_PLAN §6.2).

Primary model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
(multilingual). Fallback (recorded in the report): microsoft/deberta-large-mnli
(English only). Residency: kept on GPU between evals — the M2 envelope left
7.4 GiB headroom and the model is ~0.4 GiB fp16 (M3 decision per spec item 6).
"""

from __future__ import annotations

import torch

PRIMARY = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
FALLBACK = "microsoft/deberta-large-mnli"

_scorer = None


def get_scorer(device: str = "cuda"):
    global _scorer
    if _scorer is None:
        _scorer = NLIScorer(device)
    return _scorer


class NLIScorer:
    def __init__(self, device: str = "cuda"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        for name in (PRIMARY, FALLBACK):
            try:
                self.tok = AutoTokenizer.from_pretrained(name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    name, dtype=torch.float16).to(device).eval()
                self.name = name
                break
            except Exception as e:
                print(f"[nli] {name} unavailable ({type(e).__name__}); trying fallback")
        else:
            raise RuntimeError("no NLI model available")
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx = next(i for i, l in id2label.items() if "entail" in l)

    @torch.no_grad()
    def entail_prob(self, premises: list[str], hypotheses: list[str],
                    batch_size: int = 16) -> list[float]:
        out: list[float] = []
        for i in range(0, len(premises), batch_size):
            enc = self.tok(premises[i:i + batch_size], hypotheses[i:i + batch_size],
                           padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.float()
            out += logits.softmax(-1)[:, self.entail_idx].tolist()
        return out

    def bidirectional(self, a: list[str], b: list[str]) -> list[float]:
        """min(P(a ⊨ b), P(b ⊨ a)) per pair — both directions must hold."""
        fwd = self.entail_prob(a, b)
        bwd = self.entail_prob(b, a)
        return [min(x, y) for x, y in zip(fwd, bwd)]
