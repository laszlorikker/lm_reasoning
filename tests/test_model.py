"""M2 model tests (spec §7). GPU required for everything except the loss test."""

import copy

import pytest
import torch

from abstractnet.config import load_config
from abstractnet.modeling.losses import chunked_lm_loss

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.fixture(scope="module")
def model():
    from abstractnet.modeling.abstract_lm import AbstractLM

    cfg = load_config("configs/base.yaml")
    return AbstractLM(cfg, device="cuda")


TEXTS = {
    1: "Alice met Bob in the lobby.",
    3: "The committee approved the budget. Two members voted against it. A revision follows next month.",
    8: " ".join(f"Point number {i} concerns the annual review of section {i}." for i in range(12)),
}


# ---------------------------------------------------------------- chunked CE


def test_chunked_lm_loss_matches_naive_value_and_grad():
    torch.manual_seed(0)
    h1 = torch.randn(2, 37, 64, requires_grad=True)
    h2 = h1.detach().clone().requires_grad_(True)
    W1 = torch.randn(101, 64, requires_grad=True)
    W2 = W1.detach().clone().requires_grad_(True)
    labels = torch.randint(0, 101, (2, 37))
    labels[0, 30:] = -100
    naive = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(h1, W1).float().flatten(0, 1), labels.flatten(),
        ignore_index=-100)
    chunked = chunked_lm_loss(h2, W2, labels, chunk_size=16)
    assert torch.allclose(naive, chunked, atol=1e-6)
    naive.backward()
    chunked.backward()
    assert torch.allclose(h1.grad, h2.grad, atol=1e-6)
    assert torch.allclose(W1.grad, W2.grad, atol=1e-6)


# ------------------------------------------------------------- shapes & masks


@needs_gpu
def test_encode_shapes_ragged_k(model):
    z, zm, spans = model.encode([TEXTS[1], TEXTS[3], TEXTS[8]])
    assert z.shape[0] == 3 and z.shape[2] == model.cfg.model.d_z
    assert zm.sum(dim=1).tolist() == [1, 3, 8]
    assert [len(s) for s in spans] == [1, 3, 8]
    assert z.isfinite().all()


@needs_gpu
def test_padded_chunks_excluded_from_xattn(model):
    z, zm, _ = model.encode([TEXTS[1], TEXTS[3]])  # K=3, row 0 has 2 padded chunks
    tgt = model.tokenizer("A short target.", return_tensors="pt")["input_ids"].cuda()
    tgt = tgt.repeat(2, 1)
    lang = torch.zeros(2, dtype=torch.long, device="cuda")
    mask = torch.ones_like(tgt, dtype=torch.bool)
    # force gates off zero so cross-attention actually contributes
    saved = [w.gate.data.clone() for w in model._wrappers()]
    for w in model._wrappers():
        w.gate.data.fill_(0.5)
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            h_ref = model.decoder_hidden(tgt, mask, lang, z.clone(), zm, word_dropout=False)
            z_pert = z.clone()
            z_pert[0, 1:] += 1000.0  # only PADDED chunks of row 0 perturbed
            h_pert = model.decoder_hidden(tgt, mask, lang, z_pert, zm, word_dropout=False)
        assert torch.equal(h_ref, h_pert), "masked chunks leaked into cross-attention"
    finally:
        for w, s in zip(model._wrappers(), saved):
            w.gate.data.copy_(s)


# --------------------------------------------------- base equality (gates 0)


@needs_gpu
def test_gates_zero_equals_base_lm(model):
    from transformers import AutoModelForCausalLM

    assert all(float(w.gate) == 0.0 for w in model._wrappers()), "gates must init at zero"
    ids = model.tokenizer("The capital of France is", return_tensors="pt")["input_ids"].cuda()
    model.adapters_enabled(False)
    model.clear_z()
    model.eval()
    try:
        with torch.no_grad():
            ours = model.lm(ids, use_cache=False).logits.float().cpu()
        ref_model = AutoModelForCausalLM.from_pretrained(
            model.cfg.model.name_or_path, dtype=torch.float16,
            attn_implementation="sdpa").cuda().eval()
        with torch.no_grad():
            ref = ref_model(ids, use_cache=False).logits.float().cpu()
        del ref_model
        torch.cuda.empty_cache()
        assert torch.allclose(ours, ref, atol=1e-4), \
            f"max diff {(ours - ref).abs().max().item()}"
    finally:
        model.adapters_enabled(True)


@needs_gpu
def test_decode_loop_matches_hf_generate_at_gate_zero(model):
    model.adapters_enabled(False)
    model.eval()
    try:
        lang_idx = torch.zeros(1, dtype=torch.long, device="cuda")
        z = torch.zeros(1, 1, model.cfg.model.d_z, device="cuda")
        zm = torch.ones(1, 1, dtype=torch.bool, device="cuda")
        ours = model.decode(z, zm, "en", max_new_tokens=16)
        with torch.autocast("cuda", dtype=torch.float16), torch.no_grad():
            emb = model.lang_table(lang_idx).unsqueeze(1).half()
            model.clear_z()
            ref_ids = model.lm.generate(
                inputs_embeds=emb, max_new_tokens=16, do_sample=False,
                pad_token_id=model.tokenizer.eos_token_id)
        ref = model.tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
        assert ours == ref, f"{ours!r} != {ref!r}"
    finally:
        model.adapters_enabled(True)


# ----------------------------------------------------- dropout / labels / K


@needs_gpu
def test_word_dropout_train_only_and_never_labels(model):
    z, zm, _ = model.encode([TEXTS[3]])
    tgt = model.tokenizer("The target sentence here.", return_tensors="pt")["input_ids"].cuda()
    mask = torch.ones_like(tgt, dtype=torch.bool)
    lang = torch.zeros(1, dtype=torch.long, device="cuda")
    labels = tgt.clone()
    old_p = model.cfg.model.word_dropout
    model.cfg.model.word_dropout = 1.0
    try:
        model.train()
        torch.manual_seed(1)
        with torch.autocast("cuda", dtype=torch.float16):
            h_drop = model.decoder_hidden(tgt, mask, lang, z, zm, word_dropout=True)
            h_plain = model.decoder_hidden(tgt, mask, lang, z, zm, word_dropout=False)
        assert not torch.equal(h_drop, h_plain), "p=1.0 dropout must change the pass"
        model.eval()
        with torch.autocast("cuda", dtype=torch.float16):
            h_eval = model.decoder_hidden(tgt, mask, lang, z, zm, word_dropout=True)
            h_eval2 = model.decoder_hidden(tgt, mask, lang, z, zm, word_dropout=False)
        assert torch.equal(h_eval, h_eval2), "dropout must be train-only"
        assert torch.equal(labels, tgt), "labels tensor must never be modified"
        # hidden has T+1 positions; loss consumes hidden[:, :-1] against T labels
        assert h_drop.shape[1] == tgt.shape[1] + 1
    finally:
        model.cfg.model.word_dropout = old_p
        model.train()


# ------------------------------------------------------------ trainable set


@needs_gpu
def test_trainable_set_and_frozen_lower_half(model):
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lm.model.layers." in name:
            idx = int(name.split("lm.model.layers.")[1].split(".")[0])
            assert idx >= model.split, f"trainable param below split: {name}"
    assert not model.lm.get_input_embeddings().weight.requires_grad
    assert not model.lm.lm_head.weight.requires_grad
    report = model.trainable_report()
    print("trainable:", report)
    assert report["total"] > 0
    for group in ("lora", "xattn", "pooler", "z_proj", "lang_table"):
        assert report.get(group, 0) > 0, f"group {group} has no trainables"


# ------------------------------------------------------------- forward smoke


@needs_gpu
def test_forward_losses_on_real_shapes(model):
    from abstractnet.data.chunking import chunk_document
    from abstractnet.data.collate import PairCollator

    rows = []
    for i, (src, tgt) in enumerate([
        (TEXTS[3], "Le comité a approuvé le budget. Deux membres ont voté contre."),
        (TEXTS[1], "In the lobby, Alice ran into Bob."),
    ]):
        sd = chunk_document(src, model.tokenizer, "en")
        td = chunk_document(tgt, model.tokenizer, "fr" if i == 0 else "en")
        nd = chunk_document(src.replace("Alice", "Carol") if "Alice" in src
                            else src.replace("Two", "Three"), model.tokenizer, "en")
        rows.append({
            "id": f"t{i}", "pair_type": "paraphrase", "origin": "test",
            "lang_src": "en", "lang_tgt": "fr" if i == 0 else "en",
            "src_ids": sd.input_ids, "src_spans": [x for s in sd.spans for x in s],
            "tgt_ids": td.input_ids, "tgt_spans": [x for s in td.spans for x in s],
            "neg_ids": [nd.input_ids], "neg_spans": [[x for s in nd.spans for x in s]],
        })
    pad = model.tokenizer.eos_token_id
    batch = PairCollator(pad_id=pad)(rows)
    model.train()
    out = model(batch)
    for key in ("loss", "recon", "contrastive", "rate_kl"):
        assert torch.isfinite(out[key]).all(), f"{key} not finite"
    assert out["gates"].shape[0] == len(model.xattn_indices)
    out["loss"].backward()
    grads = [p.grad for p in model.pooler.parameters()]
    assert all(g is not None and g.isfinite().all() for g in grads)
    model.zero_grad(set_to_none=True)
