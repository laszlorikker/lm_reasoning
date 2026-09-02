#!/usr/bin/env python
"""Validation report (eval/VAL_REPORT_SPEC.md) — per-interval version (M3).

Callable from the train loop (generate(model=…)) or standalone on a checkpoint:

    python -m eval.val_report --checkpoint runs/<run>/step_XXXXXX.pt \
        [--config configs/base.yaml] [--run runs/<run>] [--milestone]

Output: runs/<run>/val/step_XXXXXX/report.html (self-contained, PNGs embedded),
the same scalars/figures to TensorBoard when a writer is given, and an append
to runs/<run>/val/history.jsonl for the over-steps overlays.

Per-interval content: loss curves + loss scale (from steps.jsonl), gate
magnitudes over steps, z variance spectrum + effective rank, invariance
histograms + AUC, z-dependence (correct / swapped / zeroed z) with
bidirectional NLI + chrF, PCA by language and by document, throughput/telemetry
(graph 8), the 32-doc panel with decodes and top-3 z-NNs, two failure tables.
Milestone extras (COMET, full source×target heatmap) land in M4.
"""

from __future__ import annotations

import argparse
import base64
import html as html_mod
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from abstractnet.modeling.losses import doc_vectors  # noqa: E402

MAX_NEW = 128  # greedy, per the confirmed cost decisions
ENC_BS = 64


# ------------------------------------------------------------------ helpers


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def img_tag(png: bytes, title: str) -> str:
    b64 = base64.b64encode(png).decode()
    return f"<h3>{html_mod.escape(title)}</h3><img src='data:image/png;base64,{b64}'/>"


def esc(s: str) -> str:
    return html_mod.escape(s or "")


def chrf(hyp: str, ref: str) -> float:
    import sacrebleu

    return sacrebleu.sentence_chrf(hyp, [ref]).score


def batched_docvecs(model, texts: list[str], langs: list[str]) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), ENC_BS):
        z, zm, _ = model.encode(texts[i:i + ENC_BS], langs[i:i + ENC_BS])
        vecs.append(doc_vectors(z, zm).cpu().numpy())
    return np.concatenate(vecs) if vecs else np.zeros((0, model.cfg.model.d_z))


# ----------------------------------------------------------------- sections


def encode_pool(model, pool: list[dict]) -> dict:
    texts = [d["text"] for d in pool]
    langs = [d["lang"] for d in pool]
    vecs = batched_docvecs(model, texts, langs)
    # z chunk statistics for the variance spectrum
    all_var = []
    for i in range(0, min(len(texts), 1024), ENC_BS):
        z, zm, _ = model.encode(texts[i:i + ENC_BS], langs[i:i + ENC_BS])
        chunks = z[zm].float().cpu().numpy()
        all_var.append(chunks)
    chunks = np.concatenate(all_var)
    var = chunks.var(axis=0)
    p = var / max(var.sum(), 1e-12)
    eff_rank = float(np.exp(-(p * np.log(np.clip(p, 1e-12, None))).sum()))
    return {"vecs": vecs, "dim_var": var, "eff_rank": eff_rank}


def invariance(model, pool, vecs) -> dict:
    pos_pairs = [(i, d["paraphrase"], d["lang"]) for i, d in enumerate(pool) if d["paraphrase"]]
    neg_pairs = [(i, n["text"], d["lang"]) for i, d in enumerate(pool)
                 for n in d["hard_negatives"][:1]]
    neg_meta = [(n.get("kind"), n.get("rule")) for d in pool
                for n in d["hard_negatives"][:1]]
    para_vecs = batched_docvecs(model, [p[1] for p in pos_pairs], [p[2] for p in pos_pairs])
    neg_vecs = batched_docvecs(model, [p[1] for p in neg_pairs], [p[2] for p in neg_pairs])
    cos_pos = np.array([float(vecs[i] @ v) for (i, _, _), v in zip(pos_pairs, para_vecs)])
    cos_neg = np.array([float(vecs[i] @ v) for (i, _, _), v in zip(neg_pairs, neg_vecs)])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(vecs))
    cos_rand = np.array([float(vecs[i] @ vecs[j]) for i, j in enumerate(perm) if i != j])
    neg_owner = [i for (i, _, _) in neg_pairs]
    return {"cos_pos": cos_pos, "cos_neg": cos_neg, "cos_rand": cos_rand,
            "auc_hard": auc(cos_pos, cos_neg), "auc_rand": auc(cos_pos, cos_rand),
            "neg_owner": neg_owner, "neg_vecs": neg_vecs, "neg_meta": neg_meta}


def panel_decodes(model, panel: list[dict]) -> list[dict]:
    """Per panel doc: decodes with correct / swapped / zeroed z (same language),
    plus cross-lingual decodes. All decoding is batched: per source-language
    group for the three variants, per TARGET language for cross-lingual (the
    first smoke ran cross-lingual at B=1 — 64 calls, ~26 min; this groups them
    into <=6 calls)."""
    K_MAX = model.cfg.model.k_max
    results = [dict(d) for d in panel]
    z_of: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    by_lang: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(panel):
        by_lang[d["lang"]].append(i)
    for lang, idxs in by_lang.items():
        texts = [panel[i]["text"] for i in idxs]
        z, zm, _ = model.encode(texts, [lang] * len(texts))
        for j, i in enumerate(idxs):  # store per-doc z padded to K_MAX for regrouping
            pad = K_MAX - z.shape[1]
            z_of[i] = (torch.nn.functional.pad(z[j: j + 1], (0, 0, 0, pad)),
                       torch.nn.functional.pad(zm[j: j + 1], (0, pad)))
        variants = {
            "correct": (z, zm),
            "swapped": (torch.roll(z, 1, dims=0), torch.roll(zm, 1, dims=0)),
            "zeroed": (torch.zeros_like(z), zm),
        }
        for name, (zv, zmv) in variants.items():
            outs = model.decode(zv, zmv, lang, max_new_tokens=MAX_NEW)
            for i, text in zip(idxs, outs):
                results[i][f"decode_{name}"] = text
    # cross-lingual, grouped by TARGET language: one batched decode per language
    jobs: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(results):
        refs = r.get("translations") or {}
        others = [l for l in refs if l != r["lang"]][:2]
        others += [l for l in ("en", "fr") if l != r["lang"] and l not in others]
        r["xling"] = {}
        for tl in others[:2]:
            jobs[tl].append(i)
    for tl, idxs in jobs.items():
        zs = torch.cat([z_of[i][0] for i in idxs])
        zms = torch.cat([z_of[i][1] for i in idxs])
        outs = model.decode(zs, zms, tl, max_new_tokens=MAX_NEW)
        for i, out in zip(idxs, outs):
            refs = results[i].get("translations") or {}
            results[i]["xling"][tl] = {
                "decode": out,
                "chrf_vs_ref": round(chrf(out, refs[tl]), 1) if tl in refs else None,
            }
    return results


def score_panel(results: list[dict]) -> dict:
    from eval.nli import get_scorer

    nli = get_scorer()
    agg: dict[str, list[float]] = defaultdict(list)
    for cond in ("correct", "swapped", "zeroed"):
        pairs_a = [r["text"] for r in results]
        pairs_b = [r[f"decode_{cond}"] for r in results]
        scores = nli.bidirectional(pairs_a, pairs_b)
        for r, s in zip(results, scores):
            r[f"nli_{cond}"] = round(s, 3)
            r[f"chrf_{cond}"] = round(chrf(r[f"decode_{cond}"], r["text"]), 1)
            agg[f"nli_{cond}"].append(s)
            agg[f"chrf_{cond}"].append(r[f"chrf_{cond}"])
    return {k: float(np.mean(v)) for k, v in agg.items()} | {"nli_model": nli.name}


def nearest_neighbours(vecs: np.ndarray, panel_vecs: np.ndarray, pool: list[dict], k=3):
    sims = panel_vecs @ vecs.T
    out = []
    for row in sims:
        idx = np.argsort(-row)[:k]
        out.append([(pool[j]["id"], round(float(row[j]), 3)) for j in idx])
    return out


# ------------------------------------------------------------------- plots


def make_plots(run_dir: Path | None, step: int, enc: dict, inv: dict,
               panel_scores: dict, history: list[dict]) -> list[tuple[str, bytes]]:
    plots: list[tuple[str, bytes]] = []

    if run_dir and (run_dir / "steps.jsonl").exists():
        steps, recon, con, kl, scale, toks = [], [], [], [], [], []
        with open(run_dir / "steps.jsonl") as f:
            for line in f:
                r = json.loads(line)
                steps.append(r["step"]); recon.append(r["recon"]); con.append(r["con"])
                kl.append(r["kl"]); scale.append(r["scale"]); toks.append(r.get("tok_s", 0))
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        ax[0].plot(steps, recon, label="recon"); ax[0].plot(steps, con, label="contrastive")
        ax[0].plot(steps, kl, label="rate_kl"); ax[0].set_yscale("log")
        ax[0].legend(); ax[0].set_title("train loss terms"); ax[0].set_xlabel("step")
        ax[1].plot(steps, scale); ax[1].set_yscale("log", base=2)
        ax[1].set_title("GradScaler loss scale (fp16 health)"); ax[1].set_xlabel("step")
        plots.append(("1. Loss terms and loss scale", fig_png(fig)))

        if (run_dir / "telemetry.csv").exists():
            t, clock, power, temp = [], [], [], []
            with open(run_dir / "telemetry.csv") as f:
                next(f)
                for line in f:
                    try:
                        a, b, c, d, _ = line.strip().split(",")
                        t.append(float(a)); clock.append(float(b))
                        power.append(float(c)); temp.append(float(d))
                    except ValueError:
                        continue
            if t:
                t0 = t[0]
                fig, ax1 = plt.subplots(figsize=(11, 3.5))
                ax1.plot([(x - t0) / 60 for x in t], clock, color="tab:blue", label="SM MHz")
                ax1.set_xlabel("minutes"); ax1.set_ylabel("SM MHz", color="tab:blue")
                ax2 = ax1.twinx()
                ax2.plot([(x - t0) / 60 for x in t], temp, color="tab:red", label="°C")
                ax2.plot([(x - t0) / 60 for x in t], power, color="tab:orange", label="W")
                ax2.set_ylabel("°C / W")
                ax1.set_title("8. GPU clock, power, temperature over wall clock")
                plots.append(("8. Telemetry", fig_png(fig)))

    if history:
        hs = [h["step"] for h in history]
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.2))
        ax[0].plot(hs, [h["auc_hard"] for h in history], marker="o", label="vs hard neg")
        ax[0].plot(hs, [h["auc_rand"] for h in history], marker="o", label="vs random")
        ax[0].set_title("4. invariance AUC"); ax[0].legend(); ax[0].set_ylim(0.3, 1.02)
        for cond in ("correct", "swapped", "zeroed"):
            ax[1].plot(hs, [h[f"nli_{cond}"] for h in history], marker="o", label=cond)
        ax[1].set_title("5. z-dependence (bidirectional NLI)"); ax[1].legend()
        ax[2].plot(hs, [h["eff_rank"] for h in history], marker="o")
        ax[2].set_title("3. z effective rank")
        for a in ax:
            a.set_xlabel("step")
        plots.append(("Over-steps: AUC, z-dependence, effective rank", fig_png(fig)))
        gates_hist = [h.get("gates") for h in history if h.get("gates")]
        if gates_hist:
            fig, ax = plt.subplots(figsize=(7, 3.2))
            arr = np.array([[abs(g) for g in row] for row in gates_hist])
            for j in range(arr.shape[1]):
                ax.plot([h["step"] for h in history if h.get("gates")], arr[:, j],
                        label=f"L{14 + 2 * j}")
            ax.set_title("2. |gate| per cross-attention block"); ax.legend(fontsize=7)
            ax.set_xlabel("step")
            plots.append(("2. Gate magnitudes", fig_png(fig)))

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(inv["cos_rand"], bins=50, alpha=0.5, density=True, label="random")
    ax.hist(inv["cos_neg"], bins=50, alpha=0.6, density=True, label="hard negative")
    ax.hist(inv["cos_pos"], bins=50, alpha=0.6, density=True, label="paraphrase")
    ax.set_title(f"4. z-cosine — AUC hard {inv['auc_hard']:.3f}, random {inv['auc_rand']:.3f}")
    ax.legend()
    plots.append(("4. Invariance histograms", fig_png(fig)))

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(np.sort(enc["dim_var"])[::-1])
    ax.set_yscale("log")
    ax.set_title(f"3. z per-dim variance (sorted) — eff. rank {enc['eff_rank']:.0f}/{len(enc['dim_var'])}")
    plots.append(("3. z variance spectrum", fig_png(fig)))
    return plots


def pca_plots(model, pool, vecs) -> list[tuple[str, bytes]]:
    centred = vecs - vecs.mean(0)
    _, _, vt = np.linalg.svd(centred[:1500], full_matrices=False)
    xy = centred @ vt[:2].T
    langs = sorted({d["lang"] for d in pool})
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for lang in langs:
        idx = [i for i, d in enumerate(pool) if d["lang"] == lang]
        ax.scatter(xy[idx, 0], xy[idx, 1], s=6, alpha=0.6, label=lang)
    ax.legend(); ax.set_title("6. PCA of document z — by language")
    out = [("6a. PCA by language", fig_png(fig))]

    pairs = [(i, l, t) for i, d in enumerate(pool) for l, t in (d.get("translations") or {}).items()][:120]
    if pairs:
        tvecs = batched_docvecs(model, [t for _, _, t in pairs], [l for _, l, _ in pairs])
        txy = (tvecs - vecs.mean(0)) @ vt[:2].T
        fig, ax = plt.subplots(figsize=(6.5, 5))
        cmap = plt.get_cmap("tab20")
        for j, (i, _, _) in enumerate(pairs):
            c = cmap(j % 20)
            ax.plot([xy[i, 0], txy[j, 0]], [xy[i, 1], txy[j, 1]], color=c, alpha=0.35, lw=0.7)
            ax.scatter([xy[i, 0], txy[j, 0]], [xy[i, 1], txy[j, 1]], color=[c], s=8)
        ax.set_title("6b. PCA — translation pairs joined (should overlap)")
        out.append(("6b. PCA by document (translation pairs)", fig_png(fig)))
    return out


# --------------------------------------------------------------------- html


def render_html(out_dir: Path, step: int, plots, panel_results, nn, inv, pool,
                panel_scores, timings, spans_of) -> Path:
    rows = []
    for r, neighbours in zip(panel_results, nn):
        marked = esc(" ⎮ ".join(spans_of(r["text"], r["lang"])))
        x = "".join(f"<div><b>→{l}:</b> {esc(v['decode'])}"
                    + (f" <i>(chrF vs ref {v['chrf_vs_ref']})</i>" if v["chrf_vs_ref"] is not None else "")
                    + "</div>" for l, v in r.get("xling", {}).items())
        rows.append(f"""
<details><summary><b>{r['id']}</b> [{r['lang']}, {r.get('role', '')}] NLI c/s/z =
 {r['nli_correct']:.2f} / {r['nli_swapped']:.2f} / {r['nli_zeroed']:.2f}</summary>
<table class='doc'>
<tr><td>source (chunks)</td><td>{marked}</td></tr>
<tr><td>target paraphrase</td><td>{esc(r.get('paraphrase') or '—')}</td></tr>
<tr><td>decode correct z</td><td>{esc(r['decode_correct'])} <i>(NLI {r['nli_correct']}, chrF {r['chrf_correct']})</i></td></tr>
<tr><td>decode swapped z</td><td>{esc(r['decode_swapped'])} <i>(NLI {r['nli_swapped']}, chrF {r['chrf_swapped']})</i></td></tr>
<tr><td>decode zeroed z</td><td>{esc(r['decode_zeroed'])} <i>(NLI {r['nli_zeroed']}, chrF {r['chrf_zeroed']})</i></td></tr>
<tr><td>cross-lingual</td><td>{x or '—'}</td></tr>
<tr><td>top-3 z-NN</td><td>{esc(str(neighbours))}</td></tr>
</table></details>""")

    worst = sorted(panel_results, key=lambda r: r["nli_correct"])[:10]
    fail1 = "".join(
        f"<tr><td>{r['id']}</td><td>{esc(str(r.get('role', '')))}</td>"
        f"<td>{r.get('k_est', '?')}</td><td>{r['nli_correct']}</td>"
        f"<td>{esc(r['text'][:140])}</td><td>{esc(r['decode_correct'][:140])}</td></tr>"
        for r in worst)
    order = np.argsort(-np.array([float(v) for v in
                                  (inv["cos_neg"] if len(inv["cos_neg"]) else [0.0])]))[:10]
    fail2 = ""
    for j in order:
        i = inv["neg_owner"][j]
        kind, rule = inv["neg_meta"][j]
        fail2 += (f"<tr><td>{pool[i]['id']}</td><td>{esc(str(kind))}</td>"
                  f"<td>{esc(str(rule))}</td><td>{pool[i].get('k_est', '?')}</td>"
                  f"<td>{inv['cos_neg'][j]:.3f}</td>"
                  f"<td>{esc(pool[i]['text'][:140])}</td>"
                  f"<td>{esc(pool[i]['hard_negatives'][0]['text'][:140])}</td></tr>")

    html = f"""<meta charset='utf-8'><title>val report step {step}</title>
<style>body{{font-family:sans-serif;max-width:1200px;margin:auto}}
table.doc td{{border:1px solid #ccc;padding:4px;vertical-align:top}}
table.fail td{{border:1px solid #ccc;padding:3px;font-size:13px}}
img{{max-width:100%}}</style>
<h1>Validation report — step {step}</h1>
<p>panel NLI (mean): correct {panel_scores['nli_correct']:.3f} ·
swapped {panel_scores['nli_swapped']:.3f} · zeroed {panel_scores['nli_zeroed']:.3f}
&nbsp;|&nbsp; chrF: {panel_scores['chrf_correct']:.1f} / {panel_scores['chrf_swapped']:.1f}
/ {panel_scores['chrf_zeroed']:.1f} &nbsp;|&nbsp; AUC hard {inv['auc_hard']:.3f},
random {inv['auc_rand']:.3f} &nbsp;|&nbsp; NLI model: {panel_scores['nli_model']}</p>
{''.join(img_tag(png, t) for t, png in plots)}
<h2>Examples panel (32 fixed docs)</h2>{''.join(rows)}
<h2>Failure: 10 lowest-NLI reconstructions (panel)</h2>
<table class='fail'><tr><th>id</th><th>role</th><th>K</th><th>NLI</th><th>source</th><th>decode</th></tr>{fail1}</table>
<h2>Failure: 10 hard negatives closest to their source (pool)</h2>
<table class='fail'><tr><th>id</th><th>kind</th><th>rule</th><th>K</th><th>cos</th><th>source</th><th>negative</th></tr>{fail2}</table>
<p><i>timings: {esc(json.dumps(timings))}</i></p>"""
    out = out_dir / "report.html"
    out.write_text(html)
    return out


# --------------------------------------------------------------------- main


def generate(model, cfg, run_dir: Path | None, step: int, writer=None,
             milestone: bool = False) -> float:
    """Build the report; returns wall seconds (train.py uses it for the <10% rule)."""
    t_all = time.monotonic()
    model.eval()
    run_dir = Path(run_dir) if run_dir else None
    out_dir = (run_dir / "val" / f"step_{step:06d}") if run_dir else Path(f"val_step_{step:06d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    pool = [json.loads(l) for l in Path(cfg.data.val_pool_path).read_text().splitlines()]
    panel = [json.loads(l) for l in Path(cfg.data.panel_path).read_text().splitlines()]

    t = time.monotonic()
    enc = encode_pool(model, pool)
    inv = invariance(model, pool, enc["vecs"])
    timings["encode_s"] = round(time.monotonic() - t, 1)

    t = time.monotonic()
    panel_results = panel_decodes(model, panel)
    timings["decode_s"] = round(time.monotonic() - t, 1)
    t = time.monotonic()
    panel_scores = score_panel(panel_results)
    timings["nli_s"] = round(time.monotonic() - t, 1)

    panel_vecs = batched_docvecs(model, [d["text"] for d in panel], [d["lang"] for d in panel])
    nn = nearest_neighbours(enc["vecs"], panel_vecs, pool)

    gates = [float(w.gate.detach()) for w in model._wrappers()]
    hist_path = (run_dir / "val" / "history.jsonl") if run_dir else out_dir / "history.jsonl"
    entry = {"step": step, "auc_hard": inv["auc_hard"], "auc_rand": inv["auc_rand"],
             "eff_rank": enc["eff_rank"], "gates": gates, **panel_scores}
    entry = {k: v for k, v in entry.items() if not isinstance(v, str)}
    history = []
    if hist_path.exists():
        history = [json.loads(l) for l in hist_path.read_text().splitlines()]
    history = [h for h in history if h["step"] != step] + [entry]
    history.sort(key=lambda h: h["step"])
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text("\n".join(json.dumps(h) for h in history) + "\n")

    t = time.monotonic()
    plots = make_plots(run_dir, step, enc, inv, panel_scores, history)
    plots += pca_plots(model, pool, enc["vecs"])
    timings["plots_s"] = round(time.monotonic() - t, 1)

    def spans_of(text: str, lang: str) -> list[str]:
        from abstractnet.data.chunking import chunk_document

        d = chunk_document(text, model.tokenizer, lang)
        return [model.tokenizer.decode(d.input_ids[s:e]) for s, e in d.spans] if d else [text]

    timings["total_s"] = round(time.monotonic() - t_all, 1)
    out = render_html(out_dir, step, plots, panel_results, nn, inv, pool,
                      panel_scores, timings, spans_of)

    if writer is not None:
        writer.add_scalar("val/auc_hard", inv["auc_hard"], step)
        writer.add_scalar("val/auc_random", inv["auc_rand"], step)
        writer.add_scalar("val/eff_rank", enc["eff_rank"], step)
        for cond in ("correct", "swapped", "zeroed"):
            writer.add_scalar(f"val/nli_{cond}", panel_scores[f"nli_{cond}"], step)
            writer.add_scalar(f"val/chrf_{cond}", panel_scores[f"chrf_{cond}"], step)

    print(f"[val] step {step}: AUC hard {inv['auc_hard']:.3f} rand {inv['auc_rand']:.3f} | "
          f"NLI c/s/z {panel_scores['nli_correct']:.2f}/{panel_scores['nli_swapped']:.2f}/"
          f"{panel_scores['nli_zeroed']:.2f} | eff rank {enc['eff_rank']:.0f} | "
          f"{timings['total_s']}s -> {out}")
    return timings["total_s"]


def mini_generate(model, cfg, run_dir, step: int, writer=None, n_docs: int = 256) -> None:
    """M3.1: ~30 s health snapshot for the first mini_report_until steps —
    z variance spectrum + effective rank + mini invariance AUC on a fixed
    seeded pool subsample + gate magnitudes. PNG + TB + history_mini.jsonl."""
    t0 = time.monotonic()
    model.eval()
    pool = [json.loads(l) for l in Path(cfg.data.val_pool_path).read_text().splitlines()]
    rng = np.random.default_rng(5)
    docs = [pool[i] for i in rng.choice(len(pool), size=min(n_docs, len(pool)), replace=False)]
    vecs = batched_docvecs(model, [d["text"] for d in docs], [d["lang"] for d in docs])
    z, zm, _ = model.encode([d["text"] for d in docs[:128]], [d["lang"] for d in docs[:128]])
    var = z[zm].float().cpu().numpy().var(axis=0)
    p = var / max(var.sum(), 1e-12)
    eff = float(np.exp(-(p * np.log(np.clip(p, 1e-12, None))).sum()))
    pos = [(i, d) for i, d in enumerate(docs) if d["paraphrase"]][:128]
    negs = [(i, d) for i, d in enumerate(docs) if d["hard_negatives"]][:128]
    pv = batched_docvecs(model, [d["paraphrase"] for _, d in pos], [d["lang"] for _, d in pos])
    nv = batched_docvecs(model, [d["hard_negatives"][0]["text"] for _, d in negs],
                         [d["lang"] for _, d in negs])
    cp = np.array([float(vecs[i] @ v) for (i, _), v in zip(pos, pv)])
    cn = np.array([float(vecs[i] @ v) for (i, _), v in zip(negs, nv)])
    a = auc(cp, cn) if len(cp) and len(cn) else float("nan")
    gates = [abs(float(w.gate.detach())) for w in model._wrappers()]

    out_dir = Path(run_dir) / "val" / "mini"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].plot(np.sort(var)[::-1])
    ax[0].set_yscale("log")
    ax[0].set_title(f"z var — eff rank {eff:.0f}")
    ax[1].hist(cn, bins=30, alpha=0.6, density=True, label="hard neg")
    ax[1].hist(cp, bins=30, alpha=0.6, density=True, label="paraphrase")
    ax[1].legend()
    ax[1].set_title(f"mini AUC {a:.3f}")
    fig.savefig(out_dir / f"step_{step:06d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    with open(out_dir / "history_mini.jsonl", "a") as f:
        f.write(json.dumps({"step": step, "eff_rank": round(eff, 1),
                            "auc_hard_mini": round(a, 4),
                            "gate_absmax": round(max(gates), 5)}) + "\n")
    if writer is not None:
        writer.add_scalar("val_mini/eff_rank", eff, step)
        writer.add_scalar("val_mini/auc_hard", a, step)
        writer.add_scalar("val_mini/gate_absmax", max(gates), step)
    print(f"[mini] step {step}: AUC {a:.3f}, eff rank {eff:.0f}, "
          f"gate|max| {max(gates):.4f}, {time.monotonic() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--run", default=None, help="run dir for history/telemetry plots")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--milestone", action="store_true")
    ap.add_argument("--pool", default=None, help="override cfg.data.val_pool_path")
    args = ap.parse_args()

    from abstractnet.config import load_config
    from abstractnet.modeling.abstract_lm import AbstractLM

    cfg = load_config(args.config)
    if args.pool:
        cfg.data.val_pool_path = args.pool
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = AbstractLM(cfg)
    state = payload.get("trainable", payload.get("state"))
    model.load_state_dict(state, strict=False)
    step = args.step if args.step is not None else payload.get("step", 0)
    run = Path(args.run) if args.run else Path(args.checkpoint).parent
    generate(model, cfg, run, step, milestone=args.milestone)


if __name__ == "__main__":
    main()
