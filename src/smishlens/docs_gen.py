"""Generate README.md, MODEL_CARD.md and docs/SLIDES_OUTLINE.md from the JSONs.

Every numeric value in the three documents is read from ``reports/results/*.json``
or from the text-free ``split_manifest.csv``. Prose is templated here; nothing is
hand-typed. No message text is read or written.
"""

from __future__ import annotations

import csv
from collections import defaultdict

from smishlens.results_summary import (
    ROOT,
    e17_table,
    evasion_source_table,
    load,
    num,
    pct,
    row,
)

MANIFEST = ROOT / "data" / "processed" / "modern" / "split_manifest.csv"

# Licence text is metadata from data/cards/, not a result number.
LICENCE = {
    "imc25": ("IMC'25 smishing", "CC BY 4.0", "train / test positives"),
    "mishra": ("Mishra & Soni", "CC BY 4.0", "LOSO fold"),
    "moz": (
        "MOZ-Smishing",
        "**unverified** (card carries a *model* licence)",
        "LOSO fold",
    ),
    "exais": ("ExAIS SMS Spam", "**unverified** (no licence file)", "FPR-only LOSO fold"),
    "smishtank": ("Smishtank Dataset I", "**unverified** (cite only)", "recall-only"),
}


def pool_stats() -> tuple[dict, int]:
    """Per-source counts and test-spam totals from the text-free manifest."""
    stats: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "pos": 0, "neg": 0, "test_spam": 0, "test_rows": 0}
    )
    with MANIFEST.open(newline="") as fh:
        for item in csv.DictReader(fh):
            src = item["source"]
            y = int(item["y"])
            stats[src]["n"] += 1
            stats[src]["pos" if y == 1 else "neg"] += 1
            if item["split"] == "test":
                stats[src]["test_rows"] += 1
                stats[src]["test_spam"] += y
    total_test_spam = sum(v["test_spam"] for v in stats.values())
    return stats, total_test_spam


def _e3() -> list[dict]:
    return load("e3_pool.json")["rows"] + load("e3_pool_mmbert_base.json")["rows"]


def _e4() -> list[dict]:
    return load("e4_loso.json")["rows"] + load("e4_loso_mmbert_small.json")["rows"]


def _gold_v2() -> dict:
    return load("e6_gold_v2_bootstrap.json")["gold"]["v2"]


def _svm_e3(source: str) -> dict:
    return row(_e3(), "svm", source)


def _svm_e4(source: str) -> dict:
    return row(_e4(), "svm", source)


def _metric_rows(rows: list[dict], model: str, sources: tuple[str, ...]) -> list[str]:
    out = []
    for source in sources:
        item = row(rows, model, source)
        if not item:
            continue
        if source == "macro":
            recall, fpr, pr = (
                item.get("macro_recall"),
                item.get("macro_fpr"),
                item.get("macro_pr_auc"),
            )
        else:
            recall, fpr, pr = item.get("recall"), item.get("fpr"), item.get("pr_auc")
        out.append(f"| `{source}` | {num(recall)} | {num(fpr)} | {num(pr)} |")
    return out


def data_table(stats: dict) -> list[str]:
    lines = [
        "| Source | Pool rows | ham / positive | Licence status | Role |",
        "|---|---:|---|---|---|",
    ]
    for key in ("imc25", "mishra", "moz", "exais", "smishtank"):
        name, licence, role = LICENCE[key]
        s = stats[key]
        lines.append(
            f"| {name} | {s['n']:,} | {s['neg']:,} / {s['pos']:,} | {licence} | {role} |"
        )
    lines.append("| UCI SMS Spam | — | — | CC BY 4.0 | Tier 1 anchor |")
    return lines


def failure_bullets() -> list[str]:
    f = load("f_experiments.json")
    diag = load("diag_language_f3.json")
    exais = f["f1_fp_by_ham_category"]["exais"]
    f3 = diag["f3"]
    rec = f3["per_fold_recovery_from_weighted_quantile"]
    moz_lang = diag["language"]["baseline_loso_fpr_by_language"]["moz"]
    exais_lang = diag["language"]["baseline_loso_fpr_by_language"]["exais"]
    shares = diag["language"]["shares"]["moz"]
    pt_share = diag["language"]["declared_shares"]["moz"]["top"]["pt"]["share"]
    moz_fpr = _svm_e4("moz")["fpr"]
    moz_recall = _svm_e4("moz")["recall"]
    moz_pr = _svm_e4("moz")["pr_auc"]
    dummy_moz = row(_e4(), "dummy", "moz")["pr_auc"]
    gold = _gold_v2()["models"]["svm"]
    recal = load("e_recal.json")["models"]["svm"]["moz"]["recal"]["100"]
    return [
        f"- **`moz` shows a language-shaped FPR gap, but the check is partial.** "
        f"Source metadata tags {pct(pt_share)} of `moz` `pt`. The stopword "
        f"heuristic (not a language-ID model) puts "
        f"{pct(shares['en']['share'])} in `en` and "
        f"{pct(shares['other']['share'])} in `other`; the `en` bucket in a 100% "
        f"`pt` source is short or ambiguous text, not real English. Ham FPR is "
        f"{num(moz_lang['other']['fpr'])} for `other`-looking versus "
        f"{num(moz_lang['en']['fpr'])} for `en`-looking rows, while `exais` "
        f"shows no gap ({num(exais_lang['en']['fpr'])} `en` versus "
        f"{num(exais_lang['other']['fpr'])} `other`). This is consistent with "
        f"language shift on `moz`, not proof of it. At the frozen threshold "
        f"`moz` recall is {num(moz_recall)} and FPR {num(moz_fpr)}; PR-AUC is "
        f"{num(moz_pr)} against a base rate of {num(dummy_moz)}.",
        f"- **ExAIS is transactional ham.** Under LOSO, transactional ham FPR is "
        f"{num(exais['transactional']['fpr'])} and bank-OTP ham FPR is "
        f"{num(exais['bank_otp']['fpr'])}. The model reads invoice and alert "
        f"language as spammy.",
        f"- **Synthetic gold false positives are legitimate bank/service "
        f"messages.** svm FPR is {num(gold['fpr']['point'])} on "
        f"{_gold_v2()['n_neg']} ham rows. The pool contains almost "
        f"no real bank-alert ham, so this is a data gap as much as a model gap.",
        f"- **Recalibration on labelled target ham does not fix `moz`.** At "
        f"k={recal['k']}, mean recall is {num(recal['recall']['mean'])} at a "
        f"realised FPR of {num(recal['fpr']['mean'])}.",
        f"- **F3 passes the rule only through collapsed recall.** On `mishra` the "
        f"collapse is the unweighted OOF quantile (a weighted quantile recovers "
        f"{num(rec['mishra']['recovered'])} recall). On `moz` the F3 PR-AUC is "
        f"{num(f3['folds']['moz']['source_class_balanced']['pr_auc'])} against a "
        f"base rate of {num(dummy_moz)} — above base rate — but recall is "
        f"{num(f3['folds']['moz']['source_class_balanced']['recall'])} at the "
        f"1%-FPR threshold: ranking weak at the top, operating point unusable. "
        f"No arm is a fix.",
        "- **Pool-wide aggregates flatter the model.** One source dominates the "
        "positives.",
    ]


def readme() -> str:
    stats, total_test_spam = pool_stats()
    base = load("e3_pool.json")["base_rate"]
    imc_share = stats["imc25"]["test_spam"] / total_test_spam
    ev = load("e7_e10_evasion.json")["models"]["svm"]
    ev_src = load("e7_e10_by_source.json")["models"]
    f = load("f_experiments.json")
    f3 = load("diag_language_f3.json")["f3"]
    e17 = load("e17_latency.json")["arms"]
    gold_meta = _gold_v2()
    gold = gold_meta["models"]
    demo = load("demo_examples_check.json")
    lines = [
        "# smishlens",
        "",
        "**How much of an SMS smishing detector survives when the messages are not",
        "from 2011?** This repository builds a modern, multi-source SMS spam/smishing",
        "pool, measures ranking and threshold portability across sources, and ships a",
        "small Streamlit triage demo on top of a frozen linear model.",
        "",
        "The short answer: **the ranking transfers better than the threshold.** A",
        "model that looks excellent in-pool degrades sharply when a whole source is",
        "held out, and the frozen 1% FPR threshold does not port. The demo is",
        "deliberately a triage aid, not a verdict.",
        "",
        "---",
        "",
        "## 1. Problem",
        "",
        "Classical SMS filters report accuracy above 98% on the UCI SMS Spam",
        "Collection, a 2011 UK/Singapore corpus with no parcel scams, no",
        "bank-impersonation lures, no Hinglish, and no modern shortener domains.",
        "Three questions follow:",
        "",
        "1. Does a model trained on a modern multi-source pool rank new messages well?",
        "2. Does its operating point (a threshold chosen for 1% FPR) transfer to a",
        "   source it has never seen?",
        "3. Can a simple, frozen display rule turn a score into a usable triage hint",
        "   without pretending to be a verdict?",
        "",
        "## 2. Data",
        "",
        "The modern pool is frozen by `data/build_modern_pool.py`: exact and",
        "near-duplicate removal, then a stratified 64/16/20 train/calibration/test",
        "split keyed by `(source, label)`. No split is committed: every",
        "processed CSV is text-bearing and git-ignored. The repository carries the",
        "text-free split manifest (`data/processed/uci/SPLIT_MANIFEST.sha256`),",
        "the source cards and the build scripts, so the splits rebuild byte for",
        "byte from the upstream releases.",
        "",
    ]
    lines += data_table(stats)
    lines += [
        "",
        "Two properties dominate every aggregate below:",
        "",
        f"- **Positive-heavy.** Pool test prevalence is "
        f"{num(base['pool_test_prevalence'])} on {base['pool_test_rows']:,} rows, so "
        f"accuracy is not reported.",
        f"- **One source dominates.** IMC'25 supplies {pct(imc_share)} of test spam. "
        f"Pool-wide evasion numbers are mostly IMC'25 numbers.",
        "",
        "Sources with unverified licences are used locally and are never",
        "redistributed; their text does not enter git. See `data/README.md` and",
        "`data/cards/`.",
        "",
        "`data/local/real_alerts.csv` is a small, local, git-ignored set of real",
        "anonymised alerts (columns `text,label`). It is excluded from the Docker",
        "image and from the public check, and it is scored once as an",
        "**evaluation set** with the frozen bundle. It is **never** used for",
        "training, calibration, threshold selection or tuning. Only aggregate",
        "counts are reported; no message text and no row id is printed, logged or",
        "committed. Run `make real-alerts` to write",
        "`reports/results/real_alerts.json` locally.",
        "",
        "## 3. Method",
        "",
        "1. **Normalise** — NFKC, strip format/zero-width characters, fold mixed-script",
        "   confusables, defang URLs and phone numbers.",
        "2. **Features** — word 1–2 gram TF-IDF + character 3–5 gram TF-IDF + a small",
        "   meta block (length, digit ratio, caps ratio, URL/shortcode/currency flags).",
        "3. **Classify** — LinearSVC (`C=0.5`, class-balanced), trained on the pool",
        "   **train** split only.",
        "4. **Threshold** — max recall subject to out-of-fold FPR ≤ 1%, computed on",
        "   train OOF scores, never on test.",
        "5. **Calibrate** — Platt (sigmoid) fitted on `cal_a`.",
        "6. **Conformal** — class-conditional (Mondrian) `q_hat` on the same `cal_a`,",
        "   `alpha = 0.05`.",
        "7. **Display** — three bands derived only from the conformal set:",
        "   `{1}` → \"likely scam\", `{0}` → \"looks ordinary\", `{0,1}` or `{}` →",
        "   \"needs review\". No new thresholds in the app.",
        "",
        "No test row and no gold row is used to fit, calibrate, select or tune",
        "anything. The synthetic gold sets are used once, for evaluation.",
        "",
        "## 4. Results",
        "",
        "All numbers are generated from `reports/results/*.json` by `make summary`",
        "→ `docs/RESULTS_SUMMARY.md`. The headline tables:",
        "",
        "### In-pool test (E3, frozen threshold)",
        "",
        "| scope | recall | FPR | PR-AUC |",
        "|---|---:|---:|---:|",
    ]
    lines += _metric_rows(_e3(), "svm", ("macro", "imc25", "smishtank", "exais"))
    lines += [
        "",
        "`macro` is the unweighted mean over `mishra` + `moz`; `exais` is FPR-only,",
        "`imc25` and `smishtank` are recall-only.",
        "",
        "### Leave-one-source-out (E4, frozen threshold)",
        "",
        "| held out | recall | FPR | PR-AUC |",
        "|---|---:|---:|---:|",
    ]
    lines += _metric_rows(_e4(), "svm", ("mishra", "moz", "exais", "imc25", "smishtank", "macro"))
    lines += [
        "",
        f"The gap between the two tables is the whole story: **ranking** mostly "
        f"holds (`mishra` PR-AUC {num(_svm_e4('mishra')['pr_auc'])}), **threshold** "
        f"portability does not (`moz` FPR {num(_svm_e4('moz')['fpr'])}).",
        "",
        f"### Synthetic gold v2 (Claude-authored, {gold_meta['n_pos']} spam / "
        f"{gold_meta['n_neg']} ham)",
        "",
        "| model | recall | FPR | PR-AUC |",
        "|---|---:|---:|---:|",
    ]
    for model, label in (("svm", "svm"), ("mmbert_base", "mmBERT-base"), ("dummy", "dummy (prior)")):
        m = gold[model]
        lines.append(
            f"| {label} | {num(m['recall']['point'])} | {num(m['fpr']['point'])} | "
            f"{num(m['pr_auc']['point'])} |"
        )
    lines += [
        "",
        f"The intervals are wide ({gold_meta['n_neg']} ham rows) and the absolute "
        f"FPR is not a field estimate. Gold is synthetic; it is never used for "
        f"selection.",
        "",
        "### Evasion (E7/E8, pool test, frozen models)",
        "",
        f"Normaliser OFF → ON, svm: clean recall "
        f"{num(ev['baseline']['off']['recall'])} → {num(ev['baseline']['on']['recall'])}, "
        f"FPR {num(ev['baseline']['off']['fpr'])} → {num(ev['baseline']['on']['fpr'])}. "
        f"Leetspeak is the costliest attack "
        f"({num(ev['attacks']['off']['leet']['recall_mean'])} → "
        f"{num(ev['attacks']['on']['leet']['recall_mean'])}).",
        "",
        "| model | source | n spam | clean off | clean on | leet off | leet on |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model, label in (("svm", "svm"), ("mmbert_base", "mmBERT-base")):
        for src in ("imc25", "mishra", "moz", "smishtank"):
            r = ev_src[model]["sources"][src]
            lines.append(
                f"| {label} | {src} | {r['n_spam']:,} | "
                f"{num(r['baseline']['off'])} | {num(r['baseline']['on'])} | "
                f"{num(r['attacks']['off']['leet']['recall_mean'])} | "
                f"{num(r['attacks']['on']['leet']['recall_mean'])} |"
            )
    lines += [
        "",
        "### Fix attempts (F1–F5)",
        "",
        "A pre-registered rule required macro LOSO FPR to fall by at least 0.05",
        "while macro PR-AUC fell by at most 0.02. **No arm is a fix.**",
        "",
        "- `prevalence_*` makes both FPR and PR-AUC worse.",
        f"- `source_class_balanced` (F3) meets the letter of the rule (FPR "
        f"{num(f['arms']['baseline']['macro']['macro_fpr'])} → "
        f"{num(f['arms']['source_class_balanced']['macro']['macro_fpr'])}) only by "
        f"collapsing macro recall to "
        f"{num(f['arms']['source_class_balanced']['macro']['macro_recall'])}. On "
        f"`mishra` that is an unweighted-quantile artifact. On `moz` the F3 "
        f"PR-AUC is {num(f3['folds']['moz']['source_class_balanced']['pr_auc'])} "
        f"against a base rate of "
        f"{num(row(_e4(), 'dummy', 'moz')['pr_auc'])} — above base rate — but "
        f"recall is {num(f3['folds']['moz']['source_class_balanced']['recall'])} "
        f"at the 1%-FPR threshold: ranking weak at the top, operating point "
        f"unusable.",
        f"- `format_masked` removes a format shortcut but worsens LOSO "
        f"(PR-AUC {num(f['arms']['baseline']['macro']['macro_pr_auc'])} → "
        f"{num(f['arms']['format_masked']['macro']['macro_pr_auc'])}).",
        "",
        "## 5. Failure analysis",
        "",
    ]
    lines += failure_bullets()
    lines += [
        "",
        "## 6. Limitations",
        "",
        "- Positive-heavy synthetic pool; not a field sample.",
        "- No real bank-alert ham, and almost no multilingual coverage.",
        "- One message at a time; no conversation context.",
        "- Threshold does not port across sources; ranking does not always either.",
        f"- The normaliser's benefit is bought with a higher FPR "
        f"({num(ev['baseline']['off']['fpr'])} → {num(ev['baseline']['on']['fpr'])} "
        f"on svm).",
        "- mmBERT-base is measured but not hosted; the hosted path is svm only.",
        "- Accuracy is never reported; the base rate would dominate it.",
        "",
        "## 7. The demo app",
        "",
        "`streamlit run app/app.py`. `requirements.txt` at the repository root is",
        "the app-minimal install; Streamlit Community Cloud reads that file. The app",
        "shows:",
        "",
        "- a calibrated P(spam), the conformal set, and one band;",
        "- extracted URL(s) and basic message stats;",
        "- an optional sender-ID field that is displayed only and **does not change",
        "  the score** (experimental);",
        f"- an always-visible notice: triage aid, not a verdict; about "
        f"{pct(gold['svm']['fpr']['point'], 0)} FPR on synthetic bank/service ham; "
        f"almost no real bank-alert ham in the pool; weak on non-English held-out "
        f"sources; nothing stored or logged;",
        f"- {demo['n_examples']} invented synthetic demo messages, verified to "
        f"have no near-duplicate in gold or the pool: "
        f"{demo['pool']['n_pairs']} pairs at the 0.90 char-TF-IDF rule, maximum "
        f"similarity {num(demo['pool']['max_sim'], 2)} against the pool and "
        f"{num(demo['gold']['max_sim'], 2)} against gold.",
        "",
        "<!-- Screenshot placeholders: replace with real captures before publishing. -->",
        "",
        "| Screenshot | File |",
        "|---|---|",
        "| Empty state with the notice | `docs/img/app-empty.png` *(placeholder)* |",
        "| Scored smishing example | `docs/img/app-scam.png` *(placeholder)* |",
        "| \"Needs review\" example | `docs/img/app-review.png` *(placeholder)* |",
        "",
        "mmBERT-base is available only behind `SMISHLENS_ENABLE_MMBERT=1` and is",
        "never required by the hosted app.",
        "",
        "## 8. How to run",
        "",
        "```bash",
        "git clone <repo> smishlens && cd smishlens",
        "make setup          # venv + deps + UCI anchor + frozen split",
        "make test           # pytest",
        "make app            # Streamlit on http://localhost:8501",
        "make summary        # regenerate all docs from the JSONs",
        "```",
        "",
        "Capstone app only (no dev tooling):",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "make build-serve                     # writes artifacts/serve/ (git-ignored)",
        "streamlit run app/app.py",
        "```",
        "",
        "The FastAPI wrapper (`make serve`) and the original UCI model remain for",
        "the Tier 1 pipeline; the capstone app uses `smishlens.serve` and the pool",
        "artifact.",
        "",
        "## 9. Repo layout",
        "",
        "```",
        "src/smishlens/     library: data, normalise, features, calibrate, conformal,",
        "                   pool_experiments, loso, f_experiments, e7_e10, serve, ...",
        "app/app.py         Streamlit capstone app (svm, frozen bands)",
        "serve/api.py       FastAPI wrapper for the Tier 1 UCI artifact",
        "data/              download/build scripts, cards/, processed splits (raw ignored)",
        "reports/           verification_log.md, results/*.json, figures/",
        "docs/              RESULTS_SUMMARY.md, SLIDES_OUTLINE.md, DEPLOYMENT.md",
        "tests/             pytest suite, including serve and Streamlit AppTest",
        "artifacts/         trained models + manifests (git-ignored; rebuilt locally)",
        "```",
        "",
        "Deployment plan: `docs/DEPLOYMENT.md` (public/private split; the public",
        "repository carries no model artifact).",
        "",
        "## 10. Licence",
        "",
        "Code is MIT (see `LICENSE`). Datasets keep their own licences. Nothing here",
        "re-licenses third-party data, and **no model trained on the unverified-licence",
        "sources is cleared for redistribution**; see `MODEL_CARD.md` and",
        "`docs/DEPLOYMENT.md`. Keeping the artifact in a private repository avoids",
        "redistributing it, but it does not clear the dataset licences for serving",
        "predictions.",
        "",
        f"_Latency (E17, CPU): svm {num(e17['svm']['batch_1_ms']['p50'], 2)} ms p50 per "
        f"message, {num(e17['svm']['disk_mb'], 1)} MB on disk._",
        "",
    ]
    return "\n".join(lines)


def model_card() -> str:
    stats, _ = pool_stats()
    ev = load("e7_e10_evasion.json")["models"]["svm"]
    gold_meta = _gold_v2()
    gold = gold_meta["models"]
    e17 = load("e17_latency.json")["arms"]
    lines = [
        "# Model card — smishlens pool svm triage",
        "",
        "## What this is",
        "",
        "A binary SMS spam/smishing **ranking model** with a frozen operating point",
        "and a class-conditional conformal display rule. It is packaged for a",
        "Streamlit triage demo, not for automated blocking.",
        "",
        "- **Model:** LinearSVC (`C=0.5`, `class_weight=\"balanced\"`, `dual=\"auto\"`, seed 42)",
        "  over word 1–2 gram + character 3–5 gram TF-IDF and a small meta block.",
        "- **Trained on:** the modern pool **train** split only.",
        "- **Calibration:** Platt (sigmoid) fitted on `cal_a`.",
        "- **Conformal:** class-conditional (Mondrian) `q_hat` fitted on the same `cal_a`,",
        "  `alpha = 0.05`.",
        "- **Frozen threshold:** `tau` from the E3 recipe (max recall s.t. out-of-fold",
        "  FPR ≤ 1%), unchanged at serving time.",
        "- **No test row and no gold row is used to fit, calibrate or select anything.**",
        "",
        "## Intended use",
        "",
        "- Research and demonstration of a triage-aid UX: score, conformal set, and one",
        "  of three display bands.",
        "- Comparing ranking quality and threshold portability across SMS sources.",
        "- Teaching material about why a high in-pool score does not transfer.",
        "",
        "## Out of scope",
        "",
        "- **Not a verdict.** The band is a triage hint. Do not auto-block, auto-delete",
        "  or auto-report a message from it.",
        f"- **Not a bank-alert filter.** The pool contains almost no real bank-alert "
        f"ham, and the model flags {pct(gold['svm']['fpr']['point'], 0)} of "
        f"legitimate bank/service messages in synthetic gold.",
        "- **Not multilingual.** The pool is overwhelmingly English; held-out",
        "  non-English sources degrade badly.",
        "- **Not a sender-reputation system.** The app shows a sender-ID field for",
        "  context only. It does not read it, and it does not change the score.",
        "- **Not a conversation model.** One message at a time, no thread context.",
        "",
        "## Data and licences",
        "",
        "Only scripts, docs and text-free manifests are committed. Every",
        "processed split is text-bearing, git-ignored and rebuilt locally with",
        "`make setup`. The UCI anchor is fetched from the upstream CC BY 4.0",
        "release and re-split at seed 42; it is not redistributed here. The",
        "synthetic gold holdout is local-only and is not distributed at all.",
        "",
    ]
    lines += data_table(stats)
    lines += [
        "",
        "MOZ, ExAIS and Smishtank have **unverified or missing data licences**.",
        "Their text is never redistributed here. A model trained on them is a derived",
        "artefact; redistributing the weights is a separate legal question and is",
        "**not cleared** by this card. The deployment split keeps the artifact in a",
        "private repository; see `docs/DEPLOYMENT.md`.",
        "",
        "## Evaluation",
        "",
        "Numbers are read from `reports/results/*.json`; see",
        "`docs/RESULTS_SUMMARY.md` for the full tables.",
        "",
        "### In-pool test (E3, frozen threshold)",
        "",
        "| scope | recall | FPR | PR-AUC |",
        "|---|---:|---:|---:|",
    ]
    lines += _metric_rows(_e3(), "svm", ("macro", "imc25", "smishtank", "exais"))
    lines += [
        "",
        f"The pool test prevalence is "
        f"{num(load('e3_pool.json')['base_rate']['pool_test_prevalence'])}, so "
        f"accuracy is not reported.",
        "",
        "### Leave-one-source-out (E4, frozen threshold)",
        "",
        "| held out | recall | FPR | PR-AUC |",
        "|---|---:|---:|---:|",
    ]
    lines += _metric_rows(_e4(), "svm", ("mishra", "moz", "exais", "imc25", "smishtank", "macro"))
    lines += [
        "",
        f"Ranking transfers better than the threshold. `moz` is the failure case: "
        f"PR-AUC {num(_svm_e4('moz')['pr_auc'])} against a base rate of "
        f"{num(row(_e4(), 'dummy', 'moz')['pr_auc'])}, i.e. near chance.",
        "",
        "### Synthetic gold v2 (Claude-authored, not field data)",
        "",
        "| model | recall | FPR | PR-AUC |",
        "|---|---:|---:|---:|",
    ]
    for model, label in (("svm", "svm"), ("mmbert_base", "mmBERT-base"), ("dummy", "dummy (prior)")):
        m = gold[model]
        lines.append(
            f"| {label} | {num(m['recall']['point'])} | {num(m['fpr']['point'])} | "
            f"{num(m['pr_auc']['point'])} |"
        )
    lines += [
        "",
        f"Only {gold_meta['n_neg']} ham rows. The FPR interval is wide, and the "
        f"absolute value is not a field estimate.",
        "",
        "### Evasion (E7/E8, pool test, frozen models)",
        "",
        f"Normaliser OFF vs ON, svm: clean recall "
        f"{num(ev['baseline']['off']['recall'])} → {num(ev['baseline']['on']['recall'])}, "
        f"FPR {num(ev['baseline']['off']['fpr'])} → {num(ev['baseline']['on']['fpr'])}. "
        f"Leetspeak is the costliest attack "
        f"({num(ev['attacks']['off']['leet']['recall_mean'])} → "
        f"{num(ev['attacks']['on']['leet']['recall_mean'])}). Pool-wide numbers are "
        f"dominated by `imc25`; the per-source breakdown is in "
        f"`docs/RESULTS_SUMMARY.md`.",
        "",
        "### Latency and footprint (E17, CPU, synthetic probes)",
        "",
        f"svm: {num(e17['svm']['batch_1_ms']['p50'], 2)} ms p50 / "
        f"{num(e17['svm']['batch_1_ms']['p95'], 2)} ms p95 per message, "
        f"{num(e17['svm']['disk_mb'], 1)} MB on disk, "
        f"{num(e17['svm']['rss_mb'], 0)} MB peak RSS. mmBERT-base: "
        f"{num(e17['mmbert_base']['batch_1_ms']['p50'], 2)} ms p50 per message, "
        f"{num(e17['mmbert_base']['disk_mb'], 1)} MB on disk, "
        f"{num(e17['mmbert_base']['rss_mb'], 0)} MB peak RSS. That is why "
        f"mmBERT-base is optional and local-only.",
        "",
        "## Known failures",
        "",
    ]
    lines += failure_bullets()
    lines += [
        "",
        "## Operational notes for the demo",
        "",
        f"- The app shows an always-visible notice: triage aid, not a verdict; about "
        f"{pct(gold['svm']['fpr']['point'], 0)} FPR on synthetic bank/service ham; "
        f"almost no real bank-alert ham in the pool; weak on non-English held-out "
        f"sources; nothing stored or logged.",
        "- The artifact is hash-manifested and refused at load time if it changed.",
        "- Bands are derived only from the frozen conformal set:",
        "  `{1}` → \"likely scam\", `{0}` → \"looks ordinary\", `{0,1}` or `{}` →",
        "  \"needs review\". No new thresholds are introduced in the app.",
        "",
        "## Maintenance",
        "",
        "- Rebuild the artifact with `python -m smishlens.serve --build` (pool train +",
        "  `cal_a` only).",
        "- Regenerate all reported numbers with `make summary`.",
        "- Any change to the recipe requires a new model card and new evaluation",
        "  numbers.",
        "",
    ]
    return "\n".join(lines)


def slides() -> str:
    stats, total_test_spam = pool_stats()
    base = load("e3_pool.json")["base_rate"]
    imc_share = stats["imc25"]["test_spam"] / total_test_spam
    ev = load("e7_e10_evasion.json")["models"]["svm"]
    ev_src = load("e7_e10_by_source.json")
    boot = load("e6_gold_v2_bootstrap.json")["gold"]["v2"]
    gold = boot["models"]
    e17 = load("e17_latency.json")["arms"]
    env = load("deploy_env.json")
    lines = [
        "# Slides outline",
        "",
        "Ten slides. Every number comes from `reports/results/*.json` and matches",
        "`docs/RESULTS_SUMMARY.md`. No message text appears on any slide.",
        "",
        "---",
        "",
        "## 1. The problem",
        "",
        "- \"98% accurate\" SMS filters are usually measured on a 2011 corpus.",
        "- How much survives when the messages are modern and from several sources?",
        "- Two claims to test separately: **ranking** quality and **threshold**",
        "  portability.",
        "- Product shape: a triage aid with three display bands, not an auto-blocker.",
        "",
        "Speaker note: state up front that the deliverable is a demo, not a verdict.",
        "",
        "---",
        "",
        "## 2. Data and licences",
        "",
        f"- Pool test: {base['pool_test_rows']:,} rows, prevalence "
        f"{num(base['pool_test_prevalence'])} (positive-heavy by design).",
        "- Sources: IMC'25, Mishra & Soni, MOZ-Smishing, ExAIS, Smishtank, UCI.",
        f"- IMC'25 supplies {pct(imc_share)} of test spam, so pool-wide numbers are "
        f"dominated by one source.",
        "- Licences: IMC'25 / Mishra / UCI CC BY 4.0; **MOZ, ExAIS and Smishtank",
        "  unverified or missing**. Raw text is never redistributed.",
        "- The pool has almost no real bank-alert ham; that gap drives the failure",
        "  mode on slide 7.",
        "",
        "Speaker note: the licence row matters for any public deployment.",
        "",
        "---",
        "",
        "## 3. Method",
        "",
        "- Normalise (NFKC, confusables, defang URLs), then word 1–2 gram + char 3–5",
        "  gram TF-IDF + meta features.",
        "- LinearSVC, class-balanced, trained on the pool **train** split.",
        "- Frozen threshold: max recall s.t. out-of-fold FPR ≤ 1%, from train OOF",
        "  only.",
        "- Platt calibration on `cal_a`; class-conditional conformal `q_hat` on the",
        "  same `cal_a`, `alpha = 0.05`.",
        "- No test or gold row is used to fit, calibrate, select or tune.",
        "",
        "---",
        "",
        "## 4. In-pool results (E3)",
        "",
        "| scope | recall | FPR | PR-AUC |",
        "|---|---:|---:|---:|",
    ]
    lines += _metric_rows(_e3(), "svm", ("macro", "imc25", "smishtank", "exais"))
    lines += [
        "",
        "- These are the flattering numbers. The next slide is the honest one.",
        "",
        "---",
        "",
        "## 5. LOSO and source shift (E4)",
        "",
        "| held out | recall | FPR | PR-AUC |",
        "|---|---:|---:|---:|",
    ]
    lines += _metric_rows(_e4(), "svm", ("mishra", "moz", "exais", "macro"))
    lines += [
        "",
        "- Ranking transfers better than the threshold.",
        f"- `moz` PR-AUC {num(_svm_e4('moz')['pr_auc'])} against a base rate of "
        f"{num(row(_e4(), 'dummy', 'moz')['pr_auc'])}: the ranking itself is near",
        f"chance there. Recall {num(_svm_e4('moz')['recall'])}, FPR "
        f"{num(_svm_e4('moz')['fpr'])}.",
        "- This is a whole-source holdout, not a random split.",
        "",
        "---",
        "",
        "## 6. Synthetic gold (Claude-authored, with bootstrap CIs)",
        "",
        "| model | recall [95% CI] | FPR [95% CI] | PR-AUC [95% CI] |",
        "|---|---|---|---|",
    ]
    for model, label in (("svm", "svm"), ("mmbert_base", "mmBERT-base"), ("dummy", "dummy (prior)")):
        m = gold[model]
        lines.append(
            f"| {label} | {num(m['recall']['point'])} "
            f"[{num(m['recall']['lo'])}, {num(m['recall']['hi'])}] | "
            f"{num(m['fpr']['point'])} [{num(m['fpr']['lo'])}, {num(m['fpr']['hi'])}] | "
            f"{num(m['pr_auc']['point'])} "
            f"[{num(m['pr_auc']['lo'])}, {num(m['pr_auc']['hi'])}] |"
        )
    lines += [
        "",
        f"- {boot['n_pos']} spam / {boot['n_neg']} ham. The intervals are wide.",
        "- Gold is synthetic, never used for selection, and its absolute FPR is not",
        "  a field estimate.",
        "",
        "---",
        "",
        "## 7. Failure analysis",
        "",
    ]
    lines += failure_bullets()
    lines += [
        "",
        "---",
        "",
        "## 8. Evasion per source (E7/E8/E10)",
        "",
        f"- Normaliser OFF → ON, svm: clean recall "
        f"{num(ev['baseline']['off']['recall'])} → {num(ev['baseline']['on']['recall'])}, "
        f"FPR {num(ev['baseline']['off']['fpr'])} → {num(ev['baseline']['on']['fpr'])}.",
        "- Pool-wide evasion is dominated by `imc25`; read the per-source table.",
    ]
    lines += evasion_source_table(ev_src)
    lines += [
        "",
        "Speaker note: never quote one evasion percentage without naming the source",
        "mix.",
        "",
        "---",
        "",
        "## 9. Latency, footprint, and the app",
        "",
    ]
    lines += e17_table(load("e17_latency.json"))
    lines += [
        "",
        "- Hosted path is svm only; mmBERT-base is optional and local-only.",
        "- Output: calibrated score, conformal set, and one band: \"likely scam\" /",
        "  \"looks ordinary\" / \"needs review\". Bands come only from the frozen",
        "  conformal rule; no new thresholds.",
        f"- Artifact on disk: {num(e17['svm']['disk_mb'], 1)} MB; peak RSS",
        f"  {num(e17['svm']['rss_mb'], 0)} MB.",
        "- Always-visible notice: triage aid, not a verdict; nothing stored or logged.",
        "",
        "---",
        "",
        "## 10. Limitations and deployment",
        "",
        "- **Limits:** positive-heavy synthetic pool; one source dominates; threshold",
        "  does not port; no real bank-alert ham; no multilingual coverage; single",
        "  message, no context; accuracy is never reported.",
        "- **Two repositories, one clone.** `origin` is public (code, docs, text-free",
        "  data). The serving bundle lives only on a local `private-deploy` branch",
        "  pushed to a `private` remote you add.",
        "- The bundle is a TF-IDF vocabulary plus SVC weights derived from pool text,",
        "  so keeping it out of the public repository avoids redistribution.",
        "- Hosted on Streamlit Community Cloud, private repository, entrypoint",
        f"  `app/app.py`, Python {env['python_version']} from `deploy_env.json`.",
        "- **Reproduce:** `make summary`, `make build-serve`,",
        "  `streamlit run app/app.py`. Full plan: `docs/DEPLOYMENT.md`.",
        "",
    ]
    return "\n".join(lines)


def demo_script() -> str:
    stats, total_test_spam = pool_stats()
    base = load("e3_pool.json")["base_rate"]
    gold = load("e6_gold_v2_bootstrap.json")["gold"]["v2"]
    e17 = load("e17_latency.json")["arms"]
    env = load("deploy_env.json")
    bands = load("demo_bands.json")
    by_label = {item["label"]: item for item in bands["rows"]}
    lines = [
        "# Demo script (3 minutes)",
        "",
        "Generated from `reports/results/*.json` by `make summary`. Use only the",
        "synthetic demo messages built into the app: they are invented for this",
        "demo, are not gold rows, and are not pool rows. Never paste a real",
        "message, and never show a real-alert evaluation row.",
        "",
        "Pre-flight: `streamlit run app/app.py`, confirm the always-visible notice",
        "is present, and keep the browser at a readable zoom.",
        "",
        "## Run order",
        "",
        "| time | action | say |",
        "|---|---|---|",
        f"| 0:00–0:20 | Open the app. Point at the notice. | \"This is a triage aid,"
        f" not a verdict. The pool is {pct(base['pool_test_prevalence'])} positive, so"
        f" we never quote accuracy.\" |",
        "| 0:20–0:50 | Click the demo button **Parcel fee**, then **Check message**. |"
        f" \"A delivery-fee lure with a defanged URL. Band: {by_label['Parcel fee']['band']};"
        f" calibrated score {num(by_label['Parcel fee']['p_spam'], 3)}. The URL is"
        " extracted for the user, not visited.\" |",
        "| 0:50–1:20 | Click **Bank OTP (legit)**. |"
        f" \"A legitimate one-time code. Band: {by_label['Bank OTP (legit)']['band']};"
        f" score {num(by_label['Bank OTP (legit)']['p_spam'], 3)}. This is the case that"
        " matters: a bank alert must not be auto-blocked.\" |",
        "| 1:20–1:45 | Click **Dinner plan**. |"
        f" \"Ordinary chat. Band: {by_label['Dinner plan']['band']} — the conformal set"
        f" is empty, so the app asks for review instead of guessing. Score"
        f" {num(by_label['Dinner plan']['p_spam'], 3)}.\" |",
        "| 1:45–2:05 | Click **Invoice**. |"
        f" \"A legitimate invoice message. Band: {by_label['Invoice']['band']}. That is"
        " the known failure: legitimate transactional text reads as spammy. We show"
        " it rather than hide it.\" |",
        "| 2:05–2:25 | Click **Hinglish KYC**. |"
        f" \"A Hinglish KYC lure. Band: {by_label['Hinglish KYC']['band']}; score"
        f" {num(by_label['Hinglish KYC']['p_spam'], 3)}. The pool is mostly English,"
        " so this is a demonstration, not a multilingual claim.\" |",
        "| 2:25–2:45 | Point at the footer recipe line. |"
        f" \"Frozen: pool train, Platt and conformal on cal_a only. Scoring takes"
        f" {num(e17['svm']['batch_1_ms']['p50'], 2)} ms on CPU; the artifact is"
        f" {num(e17['svm']['disk_mb'], 1)} MB.\" |",
        "| 2:45–3:00 | Stop. | \"The ranking transfers better than the threshold. The"
        " demo is deliberately not an auto-blocker.\" |",
        "",
        "## Likely questions",
        "",
        "- **Why no accuracy?** The pool test set is "
        f"{pct(base['pool_test_prevalence'])} positive on {base['pool_test_rows']:,}",
        "  rows, so accuracy would mostly restate the base rate. We report recall,",
        "  FPR and PR-AUC per source instead.",
        "- **Why is FPR about "
        f"{num(gold['models']['svm']['fpr']['point'], 1)} on gold?** Gold is",
        f"  synthetic and has only {gold['n_neg']} ham rows, many of them legitimate",
        "  bank/service messages. The pool has almost no real bank-alert ham, so the",
        "  model never learned that pattern. The interval is wide and the absolute",
        "  FPR is not a field estimate.",
        "- **Why is mmBERT not hosted?** It needs "
        f"{num(e17['mmbert_base']['rss_mb'], 0)} MB peak RSS and",
        f"  {num(e17['mmbert_base']['disk_mb'], 1)} MB on disk, against",
        f"  {num(e17['svm']['rss_mb'], 0)} MB and {num(e17['svm']['disk_mb'], 1)} MB",
        "  for svm, and it is optional behind `SMISHLENS_ENABLE_MMBERT=1`.",
        "- **Why is the artifact private?** The bundle is a TF-IDF vocabulary plus",
        "  SVC weights derived from pool text. MOZ, ExAIS and Smishtank have",
        "  unverified or missing text licences, so redistributing the bundle is not",
        "  clearly safe. It stays on the `private` remote; the public repository is",
        "  code and text-free data only.",
        "- **Does it generalise to new sources?** Ranking often does; the frozen",
        "  threshold does not. `moz` is the counter-example where even ranking is",
        "  near chance.",
        "- **Was gold used for tuning?** No. Gold was scored once, after the",
        "  recipe was frozen.",
        "",
        "## Numbers used here",
        "",
        "All values are read from `reports/results/*.json`:",
        "`e3_pool.json`, `e6_gold_v2_bootstrap.json`, `e17_latency.json`,",
        f"`demo_bands.json` and `deploy_env.json` (Python {env['python_version']}).",
        "`demo_bands.json` is written by `make demo-bands` from the frozen bundle;",
        "it holds label, band, conformal set and score only, never message text.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    (ROOT / "README.md").write_text(readme())
    (ROOT / "MODEL_CARD.md").write_text(model_card())
    (ROOT / "docs" / "SLIDES_OUTLINE.md").write_text(slides())
    (ROOT / "docs" / "DEMO_SCRIPT.md").write_text(demo_script())
    print("wrote README.md, MODEL_CARD.md, docs/SLIDES_OUTLINE.md, docs/DEMO_SCRIPT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
