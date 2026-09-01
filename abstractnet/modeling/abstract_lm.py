"""AbstractLM — the Phase-1 encoder/decoder pair on a split Qwen3 (M2).

One base LM, split at cfg.split_layer. The encoder runs embed_tokens +
layers[:SPLIT] (the model's own modules, causal as pretrained) and pools each
chunk to z. The decoder runs the full stack with gated cross-attention wrappers
on layers SPLIT, SPLIT+xattn_every, …; gates start at zero, so with no z set
the decoder IS the base LM. Spec: M2 kickoff (binding); readings recorded in
M2_DESIGN.md.

Right-padding note: encoder and decoder run with attention_mask=None. With
causal attention and right padding, real tokens never attend to pad positions
(pads are in the future); pad-position states are garbage but are never read
(pooling masks to spans, the loss ignores -100). This reuses the model's own
mask construction exactly as the spec requires.

Gradient-checkpointing note: the xattn wrappers read their z context at
BACKWARD time (recompute). set_z overwrites per forward and is NOT cleared
after training forwards — clearing before backward would silently drop the
xattn contribution from recomputation. clear_z() is for base-equality paths.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from abstractnet.config import Config
from abstractnet.data.chunking import chunk_document
from abstractnet.data.collate import LANGS
from abstractnet.modeling.losses import chunked_lm_loss, doc_vectors, symmetric_info_nce
from abstractnet.modeling.pooling import ChunkPooler
from abstractnet.modeling.rate import make_rate
from abstractnet.modeling.xattn import GatedXAttnLayer

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class AbstractLM(nn.Module):
    def __init__(self, cfg: Config, device: str = "cuda"):
        super().__init__()
        from peft import LoraConfig, inject_adapter_in_model

        from abstractnet.utils.load import load_base_lm

        self.cfg = cfg
        mc = cfg.model
        assert list(mc.languages) == LANGS, "config languages must match collate.LANGS order"
        self.lm, self.tokenizer = load_base_lm(mc, device=device)
        self.lm.requires_grad_(False)
        self.split = mc.split_layer
        self.n_layers = self.lm.config.num_hidden_layers
        hidden = self.lm.config.hidden_size
        self.device_ = device

        # LoRA on the DECODER half only (upper layers); assert none below split
        lora = LoraConfig(
            r=mc.lora_r, lora_alpha=mc.lora_alpha, lora_dropout=mc.lora_dropout,
            target_modules=LORA_TARGETS, bias="none",
            layers_to_transform=list(range(self.split, self.n_layers)),
            layers_pattern="layers",
        )
        self.lm = inject_adapter_in_model(lora, self.lm)
        for name, p in self.lm.named_parameters():
            if "lora_" in name:
                layer_idx = int(name.split("layers.")[1].split(".")[0])
                assert layer_idx >= self.split, f"LoRA leaked into lower half: {name}"
                p.data = p.data.float()  # fp32 masters for trainables
        if mc.encoder_lora_layers:
            raise NotImplementedError("encoder_lora_layers is an M5 ablation hook; keep empty")

        # gated cross-attention wrappers on layers SPLIT, SPLIT+every, ...
        self.xattn_indices = list(range(self.split, self.n_layers, mc.xattn_every))
        layers = self.lm.model.layers
        for i in self.xattn_indices:
            layers[i] = GatedXAttnLayer(
                layers[i], hidden, mc.xattn_heads, mc.xattn_head_dim
            ).to(device)  # wrapper params fp32, on device; wrapped layer untouched

        # new trainable modules (fp32 params; autocast handles compute dtype)
        self.pooler = ChunkPooler(hidden, mc.d_z, mc.pool_heads, mc.pool_head_dim)
        self.rate = make_rate(mc.rate, mc.d_z, mc.rate_sigma)
        self.z_proj = nn.Linear(mc.d_z, hidden, bias=False)          # shared across blocks
        self.chunk_pos = nn.Parameter(torch.randn(mc.k_max, hidden) * 0.02)  # shared
        self.lang_table = nn.Embedding(len(LANGS), hidden)
        nn.init.normal_(self.lang_table.weight, std=0.02)
        self.word_dropout_vec = nn.Parameter(torch.randn(hidden) * 0.02)
        for m in (self.pooler, self.rate, self.z_proj, self.lang_table):
            m.to(device)
        self.chunk_pos.data = self.chunk_pos.data.to(device)
        self.word_dropout_vec.data = self.word_dropout_vec.data.to(device)

        emb = self.lm.get_input_embeddings()
        assert not emb.weight.requires_grad and not self.lm.lm_head.weight.requires_grad

    # ------------------------------------------------------------- internals

    def _wrappers(self):
        return [self.lm.model.layers[i] for i in self.xattn_indices]

    def _lower_half(self, input_ids: torch.Tensor) -> torch.Tensor:
        m = self.lm.model
        h = m.embed_tokens(input_ids)
        pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        pos_emb = m.rotary_emb(h, pos)
        for layer in m.layers[: self.split]:
            out = layer(h, attention_mask=None, position_ids=pos,
                        position_embeddings=pos_emb, use_cache=False)
            h = out[0] if isinstance(out, tuple) else out
        return h

    def encode_ids(self, input_ids: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
        """input_ids [B,S] right-padded, chunk_mask [B,K,S] → clean z [B,K,d_z].

        The lower half is frozen and (with encoder_lora_layers empty) carries
        no trainable params, so it runs under no_grad: gradients reach the
        pooler and stop there by construction — exact, and it removes the
        lower-half activation cost from every train step (M2_DESIGN.md)."""
        with torch.no_grad():
            states = self._lower_half(input_ids)
        return self.pooler(states.detach(), chunk_mask)

    def _set_z(self, z: torch.Tensor, z_mask: torch.Tensor) -> None:
        K = z.shape[1]
        z_h = self.z_proj(z) + self.chunk_pos[:K].unsqueeze(0).to(z.dtype)
        for w in self._wrappers():
            w.set_z(z_h, z_mask)

    def clear_z(self) -> None:
        for w in self._wrappers():
            w.clear_z()

    def decoder_hidden(self, tgt_ids, tgt_mask, lang_idx, z, z_mask,
                       word_dropout: bool) -> torch.Tensor:
        """Teacher-forced decoder pass → hidden [B,T+1,h] (position 0 = lang token).
        hidden[:, t] predicts tgt_ids[:, t] for t in [0,T) — the lang token is
        never a label, word dropout touches input embeddings only."""
        emb = self.lm.model.embed_tokens(tgt_ids)
        if word_dropout and self.training and self.cfg.model.word_dropout > 0:
            drop = (torch.rand_like(tgt_ids, dtype=torch.float32)
                    < self.cfg.model.word_dropout) & tgt_mask
            emb = torch.where(drop.unsqueeze(-1),
                              self.word_dropout_vec.to(emb.dtype), emb)
        lang_emb = self.lang_table(lang_idx).unsqueeze(1).to(emb.dtype)
        inputs_embeds = torch.cat([lang_emb, emb], dim=1)
        if z is not None:
            self._set_z(z, z_mask)
        out = self.lm.model(inputs_embeds=inputs_embeds, use_cache=False)
        return out.last_hidden_state

    @staticmethod
    def _merge_docs(groups):
        """groups: list of (ids [n,S], chunk_mask [n,K,S]) → concatenated with
        common padding; returns (ids, chunk_mask, sizes)."""
        S = max(g[0].shape[1] for g in groups)
        K = max(g[1].shape[1] for g in groups)
        ids_out, cm_out, sizes = [], [], []
        for ids, cm in groups:
            n, s = ids.shape
            k = cm.shape[1]
            ids_out.append(F.pad(ids, (0, S - s)))
            # F.pad pads trailing dims first: (S_left, S_right, K_left, K_right)
            cm_out.append(F.pad(cm, (0, S - s, 0, K - k)))
            sizes.append(n)
        return torch.cat(ids_out), torch.cat(cm_out), sizes

    # --------------------------------------------------------------- training

    def forward(self, batch: dict) -> dict:
        """Full training step losses on a collated batch (M2 spec §5)."""
        tc = self.cfg.train
        dev = self.device_
        with torch.autocast("cuda", dtype=torch.float16):
            groups = [(batch["src_ids"].to(dev), batch["src_chunk_mask"].to(dev)),
                      (batch["tgt_ids"].to(dev), batch["tgt_chunk_mask"].to(dev))]
            has_neg = batch["neg_ids"].shape[0] > 0
            if has_neg:
                groups.append((batch["neg_ids"].to(dev), batch["neg_chunk_mask"].to(dev)))
            ids, cm, sizes = self._merge_docs(groups)
            z_all = self.encode_ids(ids, cm)
            B = sizes[0]
            src_zm = batch["src_z_mask"].to(dev)
            tgt_zm = batch["tgt_z_mask"].to(dev)
            # slice each group back to its own K (merge padded to the common max)
            z_x = z_all[:B, : src_zm.shape[1]]
            z_xp = z_all[B: 2 * B, : tgt_zm.shape[1]]
            z_neg = z_all[2 * B:] if has_neg else None
            z_clean, z_dec, kl = self.rate(z_x, src_zm, self.training)

            hidden = self.decoder_hidden(
                batch["tgt_ids"].to(dev), batch["tgt_mask"].to(dev),
                batch["lang_tgt_idx"].to(dev), z_dec, src_zm, word_dropout=True)
            l_recon = chunked_lm_loss(hidden[:, :-1], self.lm.lm_head.weight,
                                      batch["labels"].to(dev), tc.chunk_ce_size)

            va = doc_vectors(z_clean, src_zm)
            vb = doc_vectors(z_xp, tgt_zm)
            vn = None
            if has_neg:
                vn = doc_vectors(z_neg[:, : batch["neg_z_mask"].shape[1]],
                                 batch["neg_z_mask"].to(dev))
            l_con = symmetric_info_nce(va, vb, vn, tc.tau)

            loss = l_recon + tc.lambda_c * l_con + tc.lambda_r * kl
        return {"loss": loss, "recon": l_recon.detach(), "contrastive": l_con.detach(),
                "rate_kl": kl.detach(),
                "gates": torch.stack([w.gate.detach() for w in self._wrappers()])}

    # ------------------------------------------------------------ §8 interface

    @torch.no_grad()
    def encode(self, texts: list[str], langs: list[str] | None = None):
        """texts → (z [B,K,d_z] fp32 clean, z_mask [B,K] bool, spans). Chunking
        via the M1 pipeline; langs defaults to English (§8 signature extended
        with optional langs — see M2_DESIGN.md)."""
        dcfg = self.cfg.data
        langs = langs or ["en"] * len(texts)
        docs = [chunk_document(t, self.tokenizer, l, dcfg.max_chunk_tokens,
                               dcfg.max_chunks, dcfg.max_source_tokens)
                for t, l in zip(texts, langs)]
        assert all(d is not None for d in docs), "encode() got an empty document"
        S = max(len(d.input_ids) for d in docs)
        K = max(d.k for d in docs)
        ids = torch.zeros(len(docs), S, dtype=torch.long)
        cm = torch.zeros(len(docs), K, S, dtype=torch.bool)
        zm = torch.zeros(len(docs), K, dtype=torch.bool)
        for i, d in enumerate(docs):
            ids[i, : len(d.input_ids)] = torch.tensor(d.input_ids)
            for k, (s, e) in enumerate(d.spans):
                cm[i, k, s:e] = True
                zm[i, k] = True
        with torch.autocast("cuda", dtype=torch.float16):
            z = self.encode_ids(ids.to(self.device_), cm.to(self.device_))
        return z.float(), zm.to(self.device_), [d.spans for d in docs]

    @torch.no_grad()
    def decode(self, z, z_mask, lang: str, max_new_tokens: int = 256) -> list[str]:
        """Greedy decoding from z with KV cache (own loop, not HF generate)."""
        B = z.shape[0]
        lang_idx = torch.full((B,), LANGS.index(lang), device=self.device_, dtype=torch.long)
        eos = self.tokenizer.eos_token_id
        with torch.autocast("cuda", dtype=torch.float16):
            self._set_z(z.to(self.device_), z_mask.to(self.device_))
            emb = self.lang_table(lang_idx).unsqueeze(1)
            out = self.lm.model(inputs_embeds=emb.half(), use_cache=True)
            past = out.past_key_values
            tokens, finished = [], torch.zeros(B, dtype=torch.bool, device=self.device_)
            next_id = self.lm.lm_head(out.last_hidden_state[:, -1]).argmax(-1)
            eos_t = torch.full_like(next_id, eos)
            for _ in range(max_new_tokens - 1):
                tokens.append(torch.where(finished, eos_t, next_id))
                finished |= next_id == eos
                if bool(finished.all()):
                    break
                step_emb = self.lm.model.embed_tokens(next_id).unsqueeze(1)
                out = self.lm.model(inputs_embeds=step_emb, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_id = self.lm.lm_head(out.last_hidden_state[:, -1]).argmax(-1)
            else:
                tokens.append(torch.where(finished, eos_t, next_id))
        ids = torch.stack(tokens, dim=1)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def decode_logits(self, z, z_mask, lang: str, target_ids: torch.Tensor) -> dict:
        """Teacher-forced scoring of target_ids [B,T] given z → per-token
        log-probs [B,T] and mean NLL per document [B]."""
        B, T = target_ids.shape
        lang_idx = torch.full((B,), LANGS.index(lang), device=self.device_, dtype=torch.long)
        mask = torch.ones_like(target_ids, dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.float16):
            hidden = self.decoder_hidden(target_ids.to(self.device_), mask.to(self.device_),
                                         lang_idx, z.to(self.device_), z_mask.to(self.device_),
                                         word_dropout=False)
            logits = self.lm.lm_head(hidden[:, :-1])
        logprobs = logits.float().log_softmax(-1)
        tok_lp = logprobs.gather(-1, target_ids.to(self.device_).unsqueeze(-1)).squeeze(-1)
        return {"token_logprobs": tok_lp, "nll": -tok_lp.mean(dim=1)}

    # ------------------------------------------------------------ checkpoints

    def trainable_state_dict(self) -> dict:
        return {k: v for k, v in self.state_dict().items()
                if k in self._trainable_keys()}

    def _trainable_keys(self) -> set[str]:
        return {k for k, p in self.named_parameters() if p.requires_grad}

    def save_trainable(self, path: str) -> None:
        torch.save({"cfg_model": vars(self.cfg.model), "state": self.trainable_state_dict()}, path)

    @classmethod
    def load_trainable(cls, cfg: Config, path: str, device: str = "cuda") -> "AbstractLM":
        """One-call load: base model + trainable set from a checkpoint (§8)."""
        model = cls(cfg, device=device)
        payload = torch.load(path, map_location=device, weights_only=False)
        state = payload["state"]
        missing = model._trainable_keys() - set(state)
        assert not missing, f"checkpoint missing trainable keys: {sorted(missing)[:5]}…"
        model.load_state_dict(state, strict=False)
        return model

    def adapters_enabled(self, enabled: bool) -> None:
        from peft.tuners.tuners_utils import BaseTunerLayer

        for m in self.lm.modules():
            if isinstance(m, BaseTunerLayer):
                m.enable_adapters(enabled)

    def trainable_report(self) -> dict:
        by_group: dict[str, int] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            key = ("lora" if "lora_" in name else
                   "xattn" if any(f"layers.{i}." in name and "lora_" not in name
                                  for i in self.xattn_indices) else
                   name.split(".")[0])
            by_group[key] = by_group.get(key, 0) + p.numel()
        by_group["total"] = sum(by_group.values())
        return by_group
