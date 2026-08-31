#!/usr/bin/env python
"""M1 S5: build the frozen validation pool (2k docs) and the 32-document panel.

Outputs (COMMITTED as fixtures — same seed, same IDs, forever):
  data/fixtures/val_pool_v1.jsonl   ~2000 docs: text, lang, origin, k_est,
                                    paraphrase partner, translations{},
                                    hard_negatives[{text, kind, rule}]
  data/fixtures/panel_v1.jsonl      32 docs (ids panel-00..panel-31): >=1/3
                                    multi-chunk (3-8 sentences); singles are
                                    deliberately hard (negation / numbers /
                                    entities / long multi-clause), hand-written
                                    with controlled propositions
Stats -> runs/m1/val_panel_stats.json.

Val separation from train (by construction): held-out splits (paws-x test,
qqp/mnli/xnli validation, opus-100 test), the first VAL_RESERVED_DOCS documents
per concat config, math problems with is_val_problem(problem).

Usage: python scripts/build_val_panel.py [--config configs/base.yaml]
"""

import argparse
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abstractnet.config import load_config
from abstractnet.data import pairs as P
from abstractnet.data.chunking import chunk_document
from abstractnet.data.minimal_pairs import generate_minimal_pairs_bulk

FIXTURES = Path("data/fixtures")

# ----------------------------------------------------------------- hand panel
# 20 deliberately hard singles. Each: text, lang, paraphrase, minimal negative
# (exactly one proposition flipped), optional reference translations, role.
HARD_SINGLES = [
    dict(role="negation", lang="en",
         text="The board did not reject the merger; it simply never brought the proposal to a vote.",
         paraphrase="Rather than voting the merger down, the board never actually put the proposal to a vote.",
         negative="The board rejected the merger; it simply never brought the proposal to a vote.",
         translations={"fr": "Le conseil n'a pas rejeté la fusion ; il n'a simplement jamais soumis la proposition au vote.",
                       "es": "La junta no rechazó la fusión; simplemente nunca sometió la propuesta a votación."}),
    dict(role="negation", lang="en",
         text="Not all of the applicants who passed the written test were invited to the interview.",
         paraphrase="Some applicants passed the written test but still received no interview invitation.",
         negative="All of the applicants who passed the written test were invited to the interview.",
         translations={}),
    dict(role="numbers", lang="en",
         text="Revenue rose from $4.2 million in 2018 to $6.8 million in 2021, while headcount grew by only 12 percent.",
         paraphrase="Between 2018 and 2021, revenue increased from $4.2 million to $6.8 million, yet staffing expanded by a mere 12 percent.",
         negative="Revenue rose from $4.2 million in 2018 to $5.8 million in 2021, while headcount grew by only 12 percent.",
         translations={"fr": "Le chiffre d'affaires est passé de 4,2 millions de dollars en 2018 à 6,8 millions en 2021, tandis que les effectifs n'ont augmenté que de 12 pour cent."}),
    dict(role="numbers", lang="en",
         text="The bridge, completed in 1937 after four years of construction, spans 2,737 meters.",
         paraphrase="Construction of the bridge took four years and finished in 1937; its span measures 2,737 meters.",
         negative="The bridge, completed in 1939 after four years of construction, spans 2,737 meters.",
         translations={}),
    dict(role="entities", lang="en",
         text="Marie Curie left Warsaw for Paris in 1891, and it was there that she met Pierre.",
         paraphrase="In 1891 Marie Curie moved from Warsaw to Paris, where she and Pierre first met.",
         negative="Marie Curie left Paris for Warsaw in 1891, and it was there that she met Pierre.",
         translations={}),
    dict(role="entities", lang="en",
         text="Airbus, not Boeing, won the contract to supply the Australian carrier with 28 aircraft.",
         paraphrase="The contract for 28 aircraft for the Australian airline went to Airbus rather than Boeing.",
         negative="Boeing, not Airbus, won the contract to supply the Australian carrier with 28 aircraft.",
         translations={}),
    dict(role="long_clause", lang="en",
         text="Although the committee had promised, after months of delay and repeated warnings from its auditors, that the revised budget would be published before the end of March, the document appeared only in June, and even then without the annexes the auditors had explicitly requested.",
         paraphrase="Despite promising its auditors — after long delays and many warnings — a revised budget by late March, the committee published it only in June, still missing the annexes the auditors had specifically asked for.",
         negative="Although the committee had promised, after months of delay and repeated warnings from its auditors, that the revised budget would be published before the end of March, the document appeared only in June, and even then with the annexes the auditors had explicitly requested.",
         translations={}),
    dict(role="arg_structure", lang="en",
         text="The regulator fined the bank because the bank had misled the regulator's inspectors.",
         paraphrase="Because the bank misled its inspectors, the regulator imposed a fine on it.",
         negative="The bank fined the regulator because the regulator had misled the bank's inspectors.",
         translations={}),
    dict(role="negation", lang="fr",
         text="Le laboratoire n'a jamais confirmé que le vaccin était inefficace ; il a seulement signalé des données incomplètes.",
         paraphrase="Le laboratoire a uniquement fait état de données incomplètes, sans jamais confirmer l'inefficacité du vaccin.",
         negative="Le laboratoire a confirmé que le vaccin était inefficace ; il a seulement signalé des données incomplètes.",
         translations={"en": "The laboratory never confirmed that the vaccine was ineffective; it only reported incomplete data."}),
    dict(role="numbers", lang="fr",
         text="La ligne 14 transporte environ 550 000 voyageurs par jour, soit une hausse de 8 % depuis 2022.",
         paraphrase="Chaque jour, quelque 550 000 personnes empruntent la ligne 14, en progression de 8 % par rapport à 2022.",
         negative="La ligne 14 transporte environ 450 000 voyageurs par jour, soit une hausse de 8 % depuis 2022.",
         translations={"en": "Line 14 carries about 550,000 passengers a day, up 8 percent since 2022."}),
    dict(role="entities", lang="fr",
         text="C'est Lyon, et non Marseille, qui accueillera le sommet en septembre.",
         paraphrase="Le sommet de septembre se tiendra à Lyon plutôt qu'à Marseille.",
         negative="C'est Marseille, et non Lyon, qui accueillera le sommet en septembre.",
         translations={}),
    dict(role="negation", lang="de",
         text="Der Ausschuss hat den Bericht nicht abgelehnt, sondern die Abstimmung auf November verschoben.",
         paraphrase="Statt den Bericht abzulehnen, verschob der Ausschuss die Abstimmung auf November.",
         negative="Der Ausschuss hat den Bericht abgelehnt und die Abstimmung auf November verschoben.",
         translations={"en": "The committee did not reject the report; instead it postponed the vote until November."}),
    dict(role="numbers", lang="de",
         text="Die Miete stieg innerhalb von fünf Jahren von 8,50 Euro auf 12,30 Euro pro Quadratmeter.",
         paraphrase="Binnen fünf Jahren erhöhte sich der Quadratmeterpreis von 8,50 auf 12,30 Euro.",
         negative="Die Miete stieg innerhalb von fünf Jahren von 8,50 Euro auf 11,30 Euro pro Quadratmeter.",
         translations={}),
    dict(role="entities", lang="de",
         text="Nicht die Deutsche Bahn, sondern ein privater Betreiber übernimmt ab 2027 die Regionalstrecke.",
         paraphrase="Die Regionalstrecke geht 2027 an einen privaten Betreiber und nicht an die Deutsche Bahn.",
         negative="Die Deutsche Bahn, nicht ein privater Betreiber, übernimmt ab 2027 die Regionalstrecke.",
         translations={}),
    dict(role="negation", lang="es",
         text="El ayuntamiento no aprobó los 3 200 000 euros solicitados; concedió únicamente 1 800 000.",
         paraphrase="En lugar de los 3,2 millones de euros pedidos, el ayuntamiento otorgó solo 1,8 millones.",
         negative="El ayuntamiento aprobó los 3 200 000 euros solicitados; concedió únicamente 1 800 000.",
         translations={}),
    dict(role="entities", lang="es",
         text="Fue García, y no Martínez, quien firmó el acuerdo en Bogotá.",
         paraphrase="El acuerdo de Bogotá lo firmó García, no Martínez.",
         negative="Fue Martínez, y no García, quien firmó el acuerdo en Bogotá.",
         translations={}),
    dict(role="negation", lang="it",
         text="La società non ha smentito le dimissioni dell'amministratore delegato; ha soltanto rinviato ogni commento.",
         paraphrase="Anziché smentire le dimissioni dell'amministratore delegato, la società ha semplicemente rinviato ogni commento.",
         negative="La società ha smentito le dimissioni dell'amministratore delegato; ha soltanto rinviato ogni commento.",
         translations={}),
    dict(role="numbers", lang="it",
         text="Il museo ha registrato 48 500 visitatori in agosto, il 15 per cento in più rispetto a luglio.",
         paraphrase="Ad agosto i visitatori del museo sono stati 48 500, con un aumento del 15 per cento rispetto a luglio.",
         negative="Il museo ha registrato 38 500 visitatori in agosto, il 15 per cento in più rispetto a luglio.",
         translations={}),
    dict(role="negation", lang="pt",
         text="O tribunal não anulou o contrato; apenas suspendeu os pagamentos até a auditoria.",
         paraphrase="Em vez de anular o contrato, o tribunal apenas suspendeu os pagamentos até a auditoria.",
         negative="O tribunal anulou o contrato; apenas suspendeu os pagamentos até a auditoria.",
         translations={}),
    dict(role="entities", lang="pt",
         text="A fábrica de Porto Alegre produziu 12 400 unidades em março, superando a unidade de Curitiba pela primeira vez.",
         paraphrase="Em março, com 12 400 unidades, a planta de Porto Alegre ultrapassou pela primeira vez a de Curitiba.",
         negative="A fábrica de Curitiba produziu 12 400 unidades em março, superando a unidade de Porto Alegre pela primeira vez.",
         translations={}),
]


def digit_bump_negative(text: str) -> str | None:
    """Language-agnostic minimal pair: bump the first number in the document."""
    m = re.search(r"\d+", text)
    if not m:
        return None
    val = int(m.group())
    return text[: m.start()] + str(val + 1 if val % 2 else val + 3) + text[m.end():]


def k_of(text: str, lang: str, tok) -> int:
    doc = chunk_document(text, tok, lang)
    return doc.k if doc else 0


# ----------------------------------------------------------------- val pool


def pool_paws(tok, per_lang=140):
    pools = P.load_paws_pools(["en", "fr", "de", "es"], split="test")
    for lang, pool in pools.items():
        for s1, s2, adv in pool[:per_lang]:
            negs = [dict(text=adv, kind="paws_adversarial", rule=None)] if adv else []
            yield dict(text=s1, lang=lang, origin=f"paws-x-{lang}-test",
                       paraphrase=s2, translations={}, hard_negatives=negs)


def pool_opus(tok, per_cfg=100):
    import datasets

    for cfg in P.OPUS_CFGS:
        a, b = cfg.split("-")
        try:
            ds = datasets.load_dataset("Helsinki-NLP/opus-100", cfg, split="test")
        except Exception:
            ds = datasets.load_dataset("Helsinki-NLP/opus-100", cfg, split="validation")
        got = 0
        for r in ds:
            sa, sb = r["translation"][a].strip(), r["translation"][b].strip()
            if not (P._ok_sent(sa, 4, 60) and P._ok_sent(sb, 4, 60)):
                continue
            src_is_a = got % 2 == 0
            src, ls = (sa, a) if src_is_a else (sb, b)
            tgt, lt = (sb, b) if src_is_a else (sa, a)
            yield dict(text=src, lang=ls, origin=f"opus100-{cfg}-heldout",
                       paraphrase=None, translations={lt: tgt}, hard_negatives=[])
            got += 1
            if got >= per_cfg:
                break


def pool_concat(cfg, tok, n=500):
    stats = Counter()
    p = cfg.data.pilot
    src_cfg = p["sources"]["concat_translation"]
    for ex in itertools.islice(
        P.iter_concat_translation(10**9, p["seed"], src_cfg["k_min"], src_cfg["k_max"], stats, val=True), n
    ):
        yield dict(text=ex.source, lang=ex.lang_src, origin=ex.origin + "-valreserved",
                   paraphrase=None, translations={ex.lang_tgt: ex.target}, hard_negatives=[])


def pool_nli(tok, n=180):
    for ex in P.iter_nli_entailment(n, seed=23, split="validation_matched"):
        yield dict(text=ex.source, lang="en", origin="multi_nli-val",
                   paraphrase=ex.target, translations={},
                   hard_negatives=[dict(text=ex.negatives[0], kind="nli_contradiction", rule=None)])


def pool_xnli(tok, per_lang=50):
    import datasets

    ds = datasets.load_dataset("facebook/xnli", "all_languages", split="validation")
    grouped: dict[str, dict] = {}
    for r in ds:
        prem = dict(zip(r["premise"].keys(), r["premise"].values())) if isinstance(r["premise"], dict) else r["premise"]
        hyp = dict(zip(r["hypothesis"]["language"], r["hypothesis"]["translation"]))
        g = grouped.setdefault(prem["en"], {"premise": prem, "hyps": {}})
        g["hyps"].setdefault(r["label"], hyp)
    count = Counter()
    for g in grouped.values():
        if 2 not in g["hyps"]:
            continue
        for lang in ("en", "fr", "de", "es"):
            if count[lang] >= per_lang:
                continue
            text = g["premise"].get(lang, "").strip()
            if not P._ok_sent(text, 4, 90):
                continue
            para = g["hyps"].get(0, {}).get(lang)
            yield dict(text=text, lang=lang, origin="xnli-val",
                       paraphrase=para.strip() if para else None, translations={},
                       hard_negatives=[dict(text=g["hyps"][2][lang].strip(), kind="xnli_contradiction", rule=None)])
            count[lang] += 1
            break  # one language per premise: keeps premises distinct across the pool
        if sum(count.values()) >= 4 * per_lang:
            break


def pool_math(tok, n=60):
    for ex in itertools.islice(P.iter_math_derivations(10**9, seed=29, val=True), n):
        yield dict(text=ex.source, lang="en", origin=ex.origin + "-val",
                   paraphrase=ex.target, translations={}, hard_negatives=[])


# ----------------------------------------------------------------- panel


def draw_panel_multichunk(cfg, tok):
    """9 concat-translation docs (val-reserved, deterministic) + 3 math val docs."""
    docs = []
    stats = Counter()
    p = cfg.data.pilot
    sc = p["sources"]["concat_translation"]
    per_origin = Counter()
    for ex in P.iter_concat_translation(10**9, p["seed"], sc["k_min"], sc["k_max"], stats, val=True):
        if len(docs) >= 9:
            break
        if per_origin[ex.origin] >= 2:
            continue
        if k_of(ex.source, ex.lang_src, tok) < 3:
            continue
        neg = digit_bump_negative(ex.source)
        negs = [dict(text=neg, kind="minimal_pair", rule="digit_bump")] if neg else []
        docs.append(dict(role="multichunk_translation", lang=ex.lang_src, text=ex.source,
                         paraphrase=None, translations={ex.lang_tgt: ex.target},
                         hard_negatives=negs, origin=ex.origin + "-valreserved"))
        per_origin[ex.origin] += 1
    for ex in itertools.islice(P.iter_math_derivations(10**9, seed=31, val=True), 3):
        docs.append(dict(role="multichunk_math", lang="en", text=ex.source,
                         paraphrase=ex.target, translations={}, hard_negatives=[],
                         origin=ex.origin + "-val"))
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    Path("runs/m1").mkdir(parents=True, exist_ok=True)

    # ---- val pool
    pool = []
    for gen in (pool_paws(tok), pool_opus(tok), pool_concat(cfg, tok),
                pool_nli(tok), pool_xnli(tok), pool_math(tok)):
        pool.extend(gen)
    seen = set()
    pool = [d for d in pool if not (d["text"] in seen or seen.add(d["text"]))][:2000]

    # generated minimal-pair negatives for en docs that lack any hard negative
    idx = [i for i, d in enumerate(pool) if d["lang"] == "en" and not d["hard_negatives"]]
    for i, mps in zip(idx, generate_minimal_pairs_bulk([pool[i]["text"] for i in idx], n=1)):
        pool[i]["hard_negatives"] = [dict(text=m.text, kind="minimal_pair", rule=m.rule) for m in mps]

    for i, d in enumerate(pool):
        d["id"] = f"vp-{i:04d}"
        d["k_est"] = k_of(d["text"], d["lang"], tok)
        d["is_panel"] = False
    with open(FIXTURES / "val_pool_v1.jsonl", "w") as f:
        for d in pool:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ---- panel: 12 drawn multi-chunk + 20 hand-written hard singles
    panel = draw_panel_multichunk(cfg, tok)
    for h in HARD_SINGLES:
        panel.append(dict(role=h["role"], lang=h["lang"], text=h["text"],
                          paraphrase=h["paraphrase"], translations=h["translations"],
                          hard_negatives=[dict(text=h["negative"], kind="minimal_pair", rule="hand")],
                          origin="hand_written"))
    # panel math docs get generated negatives (en)
    for d in panel:
        if not d["hard_negatives"] and d["lang"] == "en":
            mps = generate_minimal_pairs_bulk([d["text"]], n=1)[0]
            d["hard_negatives"] = [dict(text=m.text, kind="minimal_pair", rule=m.rule) for m in mps]
    assert len(panel) == 32, f"panel must be exactly 32 docs, got {len(panel)}"
    for i, d in enumerate(panel):
        d["id"] = f"panel-{i:02d}"
        d["k_est"] = k_of(d["text"], d["lang"], tok)
        d["is_panel"] = True
    with open(FIXTURES / "panel_v1.jsonl", "w") as f:
        for d in panel:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ---- stats
    def summarize(docs):
        return {
            "n": len(docs),
            "langs": dict(Counter(d["lang"] for d in docs).most_common()),
            "origins": dict(Counter(d["origin"].split("-")[0] for d in docs).most_common()),
            "k_ge3": sum(d["k_est"] >= 3 for d in docs),
            "with_paraphrase": sum(bool(d["paraphrase"]) for d in docs),
            "with_translation": sum(bool(d["translations"]) for d in docs),
            "with_hard_negative": sum(bool(d["hard_negatives"]) for d in docs),
        }

    stats = {"val_pool": summarize(pool), "panel": summarize(panel)}
    Path("runs/m1/val_panel_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    assert stats["panel"]["k_ge3"] >= 11, "panel must be >=1/3 multi-chunk"
    # sentinel: streaming daemon threads can crash interpreter teardown AFTER
    # all outputs are written — judge success by this line + the files, not
    # only the exit code
    print("VAL/PANEL BUILD COMPLETE")


if __name__ == "__main__":
    main()
