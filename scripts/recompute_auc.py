#!/usr/bin/env python
"""Recompute invariance AUC from a saved checkpoint against a (fixed) pool —
built to run BESIDE the live pilot: loads only the encoder half + pooler
(~2.1 GiB fp16 vs 4.4 GiB for the full model), no NLI, small batches.

Also prints the exact removal-correction for historic evals whose checkpoints
were rotated away: the N impossible negatives sat at cos=1.000 ≥ every
positive, so they contributed zero wins and
AUC_clean = AUC_logged x n_neg / (n_neg - N)  (exact under cos_pos < 1).

Usage:
    python scripts/recompute_auc.py --checkpoint runs/pilot_v1/step_000500.pt \
        --pool data/fixtures/val_pool_v3.jsonl --run runs/pilot_v1 \
        [--correct-history 12]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abstractnet.config import load_config
from abstractnet.data.chunking import chunk_document
from abstractnet.modeling.pooling import ChunkPooler


def auc(pos, neg):
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


class SlimEncoder:
    """Encoder half + pooler only, from a training checkpoint."""

    def __init__(self, cfg, checkpoint: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        mc = cfg.model
        base = AutoModelForCausalLM.from_pretrained(
            mc.name_or_path, dtype=torch.float16, attn_implementation="sdpa",
            low_cpu_mem_usage=True)  # stays on CPU; only the lower half moves
        m = base.model
        self.embed = m.embed_tokens.to("cuda")
        self.rotary = m.rotary_emb.to("cuda")
        self.layers = torch.nn.ModuleList(m.layers[: mc.split_layer]).to("cuda")
        del base
        self.pooler = ChunkPooler(2048, mc.d_z, mc.pool_heads, mc.pool_head_dim).to("cuda")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = {k[len("pooler."):]: v for k, v in payload["trainable"].items()
                 if k.startswith("pooler.")}
        missing, unexpected = self.pooler.load_state_dict(state, strict=True), None
        self.tokenizer = AutoTokenizer.from_pretrained(mc.name_or_path)
        self.dcfg = cfg.data
        self.step = payload.get("step", -1)

    @torch.no_grad()
    def docvecs(self, texts, langs, bs: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            chunk_docs = [chunk_document(t, self.tokenizer, l, self.dcfg.max_chunk_tokens,
                                         self.dcfg.max_chunks, self.dcfg.max_source_tokens)
                          for t, l in zip(texts[i:i + bs], langs[i:i + bs])]
            S = max(len(d.input_ids) for d in chunk_docs)
            K = max(d.k for d in chunk_docs)
            ids = torch.zeros(len(chunk_docs), S, dtype=torch.long)
            cm = torch.zeros(len(chunk_docs), K, S, dtype=torch.bool)
            zm = torch.zeros(len(chunk_docs), K, dtype=torch.bool)
            for j, d in enumerate(chunk_docs):
                ids[j, : len(d.input_ids)] = torch.tensor(d.input_ids)
                for k, (s, e) in enumerate(d.spans):
                    cm[j, k, s:e] = True
                    zm[j, k] = True
            ids, cm, zm = ids.cuda(), cm.cuda(), zm.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                h = self.embed(ids)
                pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
                pe = self.rotary(h, pos)
                for layer in self.layers:
                    o = layer(h, attention_mask=None, position_ids=pos,
                              position_embeddings=pe, use_cache=False)
                    h = o[0] if isinstance(o, tuple) else o
                z = self.pooler(h, cm)
            zf = z.float()
            m = zm.unsqueeze(-1).float()
            v = (zf * m).sum(-2) / m.sum(-2).clamp(min=1e-6)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--correct-history", type=int, default=0,
                    help="N impossible negatives for the exact removal correction")
    args = ap.parse_args()
    cfg = load_config("configs/base.yaml")
    enc = SlimEncoder(cfg, args.checkpoint)
    pool = [json.loads(l) for l in Path(args.pool).read_text().splitlines()]

    vecs = enc.docvecs([d["text"] for d in pool], [d["lang"] for d in pool])
    pos_pairs = [(i, d["paraphrase"], d["lang"]) for i, d in enumerate(pool) if d["paraphrase"]]
    neg_pairs = [(i, n["text"], d["lang"]) for i, d in enumerate(pool)
                 for n in d["hard_negatives"][:1]]
    pv = enc.docvecs([p[1] for p in pos_pairs], [p[2] for p in pos_pairs])
    nv = enc.docvecs([p[1] for p in neg_pairs], [p[2] for p in neg_pairs])
    cp = np.array([float(vecs[i] @ v) for (i, _, _), v in zip(pos_pairs, pv)])
    cn = np.array([float(vecs[i] @ v) for (i, _, _), v in zip(neg_pairs, nv)])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(vecs))
    cr = np.array([float(vecs[i] @ vecs[j]) for i, j in enumerate(perm) if i != j])
    result = {"step": enc.step, "pool": args.pool, "n_pos": len(cp), "n_neg": len(cn),
              "auc_hard": round(auc(cp, cn), 4), "auc_rand": round(auc(cp, cr), 4),
              "max_cos_neg": round(float(cn.max()), 4)}
    print(json.dumps(result, indent=2))
    if args.run:
        with open(Path(args.run) / "val" / "history_clean.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

    if args.correct_history and args.run:
        hist = [json.loads(l) for l in
                (Path(args.run) / "val" / "history.jsonl").read_text().splitlines()]
        n = args.correct_history
        print("\nexact removal-correction of the logged (sick) series "
              f"(x n_neg/(n_neg-{n})):")
        for h in hist:
            n_neg = 885  # v2 pool negatives measured in the audit
            corrected = h["auc_hard"] * n_neg / (n_neg - n)
            print(f"  step {h['step']:>4}: logged {h['auc_hard']:.4f} -> "
                  f"corrected {corrected:.4f}")
    print("RECOMPUTE COMPLETE")


if __name__ == "__main__":
    main()
