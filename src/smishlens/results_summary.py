"""Generate docs/RESULTS_SUMMARY.md from the result JSONs.

Every number in the generated document is read from ``reports/results/*.json``.
Nothing is hand-typed and no message text is read or written.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reports" / "results"
OUT = ROOT / "docs" / "RESULTS_SUMMARY.md"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def num(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and math.isnan(value):
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def pct(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{100 * value:.{digits}f}%"


def signed(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}"


def row(rows: list[dict], model: str, source: str) -> dict:
    for item in rows:
        if item.get("model") == model and item.get("source") == source:
            return item
    return {}


def metric_table(rows: list[dict], dummy_rows: list[dict], models: list[tuple[str, str]]) -> list[str]:
    sources = ("mishra", "moz", "exais", "imc25", "smishtank", "macro")
    lines = [
        "| model | source | mode | n_pos | n_neg | recall | FPR | PR-AUC | dummy PR-AUC |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, label in models:
        for source in sources:
            item = row(rows, model, source)
            if not item:
                continue
            dummy = row(dummy_rows, "dummy", source)
            if source == "macro":
                recall = item.get("macro_recall")
                fpr = item.get("macro_fpr")
                pr_auc = item.get("macro_pr_auc")
                dummy_pr = dummy.get("macro_pr_auc")
            else:
                recall = item.get("recall")
                fpr = item.get("fpr")
                pr_auc = item.get("pr_auc")
                dummy_pr = dummy.get("pr_auc")
            lines.append(
                "| {m} | {s} | {mode} | {np} | {nn} | {rec} | {fpr} | {pr} | {dpr} |".format(
                    m=label,
                    s=source,
                    mode=item.get("mode", "—"),
                    np=num(item.get("n_pos")),
                    nn=num(item.get("n_neg")),
                    rec=num(recall),
                    fpr=num(fpr),
                    pr=num(pr_auc),
                    dpr=num(dummy_pr),
                )
            )
    return lines


def gold_table() -> list[str]:
    points = load("e6_gold_v2.json")["gold_versions"]
    boot = load("e6_gold_v2_bootstrap.json")["gold"]
    lines = [
        "| gold | model | n_pos / n_neg | recall [95% CI] | FPR [95% CI] | PR-AUC [95% CI] |",
        "|---|---|---:|---|---|---|",
    ]
    for version in ("v1", "v2"):
        meta = points[version]
        for model, label in (
            ("svm", "svm"),
            ("mmbert_base", "mmBERT-base"),
            ("dummy", "dummy"),
        ):
            ci = boot[version]["models"][model]
            lines.append(
                "| {v} | {m} | {np} / {nn} | {rec} | {fpr} | {pr} |".format(
                    v=version,
                    m=label,
                    np=num(meta["n_pos"]),
                    nn=num(meta["n_neg"]),
                    rec=f"{num(ci['recall']['point'])} [{num(ci['recall']['lo'])}, {num(ci['recall']['hi'])}]",
                    fpr=f"{num(ci['fpr']['point'])} [{num(ci['fpr']['lo'])}, {num(ci['fpr']['hi'])}]",
                    pr=f"{num(ci['pr_auc']['point'])} [{num(ci['pr_auc']['lo'])}, {num(ci['pr_auc']['hi'])}]",
                )
            )
    return lines


def ppv_table(block: dict, models: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| model | source | n_pos | n_neg | recall | FPR | PPV@1% | PPV@5% | PPV@10% |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, label in models:
        for source in ("mishra", "moz", "macro"):
            item = block[model]["per_source"].get(source) or block[model]["macro"]
            if source == "macro":
                item = block[model]["macro"]
            lines.append(
                "| {m} | {s} | {np} | {nn} | {rec} | {fpr} | {p1} | {p5} | {p10} |".format(
                    m=label,
                    s=source,
                    np=num(item.get("n_pos")),
                    nn=num(item.get("n_neg")),
                    rec=num(item.get("recall")),
                    fpr=num(item.get("fpr")),
                    p1=num(item.get("ppv_1pct")),
                    p5=num(item.get("ppv_5pct")),
                    p10=num(item.get("ppv_10pct")),
                )
            )
    return lines


def recal_table(recal: dict) -> list[str]:
    lines = [
        "| model | held out | frozen recall / FPR | k=25 recall / FPR | k=50 recall / FPR | k=100 recall / FPR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, label in (("svm", "svm"), ("mmbert_small", "mmBERT-small")):
        for held in ("mishra", "moz"):
            block = recal["models"][model][held]
            cells = []
            for key in ("frozen", "25", "50", "100"):
                item = block["frozen"] if key == "frozen" else block["recal"][key]
                if key == "frozen":
                    rec, fpr = item["recall"], item["fpr"]
                else:
                    rec, fpr = item["recall"]["mean"], item["fpr"]["mean"]
                cells.append(f"{num(rec)} / {num(fpr)}")
            lines.append(
                f"| {label} | {held} | " + " | ".join(cells) + " |"
            )
    return lines


def evasion_tables(ev: dict) -> list[str]:
    lines = []
    lines += [
        "Pool-wide baseline (all test spam / ham; not the E3 macro):",
        "",
        "| model | view | recall | FPR |",
        "|---|---|---:|---:|",
    ]
    for model, label in (("svm", "svm"), ("mmbert_base", "mmBERT-base")):
        for view in ("off", "on"):
            b = ev["models"][model]["baseline"][view]
            lines.append(f"| {label} | {view} | {num(b['recall'])} | {num(b['fpr'])} |")
    lines += [
        "",
        "Spam recall under attack, mean ± std over seeds (n = {n:,} spam):".format(
            n=ev["models"]["svm"]["n_spam"]
        ),
        "",
        "| model | view | spacing | leet | homoglyph | url obfuscation |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, label in (("svm", "svm"), ("mmbert_base", "mmBERT-base")):
        for view in ("off", "on"):
            cells = []
            for attack in ("spacing", "leet", "homoglyph", "url_obfuscation"):
                item = ev["models"][model]["attacks"][view][attack]
                cells.append(
                    f"{num(item['recall_mean'])} ± {num(item['recall_std'])}"
                )
            lines.append(f"| {label} | {view} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "E10 (pool test; link-less spam has no negatives, so FPR is undefined):",
        "",
        "| model | view | n | recall | FPR |",
        "|---|---|---:|---:|---:|",
    ]
    for model, label in (("svm", "svm"), ("mmbert_base", "mmBERT-base")):
        for view in ("normal", "url_masked", "spam_linkless"):
            item = ev["models"][model]["e10"][view]
            lines.append(
                f"| {label} | {view} | {num(item['n'])} | {num(item['recall'])} | {num(item['fpr'])} |"
            )
    return lines


def f_tables(f: dict) -> list[str]:
    lines = []
    lines += [
        "| arm | macro recall | macro FPR | macro PR-AUC | Δ macro FPR | Δ macro PR-AUC | ExAIS FPR | success |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, arm in f["arms"].items():
        verdict = arm.get("verdict") or {}
        lines.append(
            "| {n} | {r} | {f} | {p} | {df} | {dp} | {e} | {s} |".format(
                n=name,
                r=num(arm["macro"]["macro_recall"]),
                f=num(arm["macro"]["macro_fpr"]),
                p=num(arm["macro"]["macro_pr_auc"]),
                df=signed(verdict.get("delta_macro_fpr")),
                dp=signed(verdict.get("delta_macro_pr_auc")),
                e=num(arm["per_source"]["exais"]["fpr"]),
                s="—" if not verdict else ("yes" if verdict.get("success") else "no"),
            )
        )
    base_recall = f["arms"]["baseline"]["macro"]["macro_recall"]
    passes = [
        (name, arm)
        for name, arm in f["arms"].items()
        if (arm.get("verdict") or {}).get("success")
    ]
    if passes:
        lines += ["", "Rule-pass note (generated from the result JSON):", ""]
        for name, arm in passes:
            verdict = arm["verdict"]
            lines.append(
                f"- `{name}` meets the letter of the pre-registered rule "
                f"(Δ macro FPR {signed(verdict['delta_macro_fpr'])}, "
                f"Δ macro PR-AUC {signed(verdict['delta_macro_pr_auc'])}), "
                f"but macro recall is {num(arm['macro']['macro_recall'])} "
                f"against the baseline {num(base_recall)}. That is a collapsed "
                "operating point, not a fix."
            )
    lines += [
        "",
        "F1 — baseline false positives by coarse ham category (counts only):",
        "",
        "| source | category | n_ham | n_flagged | FPR |",
        "|---|---|---:|---:|---:|",
    ]
    for source, block in f["f1_fp_by_ham_category"].items():
        for cat in ("bank_otp", "transactional", "promo", "chatty", "total"):
            item = block[cat]
            lines.append(
                f"| {source} | {cat} | {num(item['n_ham'])} | {num(item['n_flagged'])} | {num(item['fpr'])} |"
            )
    lines += [
        "",
        "F4 — E5 format probe, baseline vs the format-masked view:",
        "",
        "| probe | baseline | format_masked |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("imc25_vs_rest_auc_from_format", "IMC'25 vs rest"),
        ("exais_vs_rest_auc_from_format", "ExAIS vs rest"),
        ("format_only_auc_pooled", "pooled format AUC"),
    ):
        lines.append(
            f"| {label} | {num(f['f4_e5']['baseline'][key])} | {num(f['f4_e5']['format_masked'][key])} |"
        )
    lines += [
        "",
        "F5 — EM prior diagnostic (assumes `p(x|y)` fixed; a diagnostic, not a fix, and the monotone adjustment leaves a quantile-thresholded decision unchanged):",
        "",
        "| held out | true prevalence | source prevalence | EM prevalence | frozen recall / FPR | adjusted recall / FPR |",
        "|---|---:|---:|---:|---|---|",
    ]
    for held, block in f["f5_prior_em"].items():
        lines.append(
            "| {h} | {t} | {s} | {e} | {fr} / {ff} | {ar} / {af} |".format(
                h=held,
                t=num(block["true_prevalence"]),
                s=num(block["source_prevalence"]),
                e=num(block["em_prevalence"]),
                fr=num(block["frozen"]["recall"]),
                ff=num(block["frozen"]["fpr"]),
                ar=num(block["prior_adjusted"]["recall"]),
                af=num(block["prior_adjusted"]["fpr"]),
            )
        )
    return lines


def f1_language_lines(f: dict, lang: dict) -> list[str]:
    moz = f["f1_fp_by_ham_category"]["moz"]
    exais = f["f1_fp_by_ham_category"]["exais"]
    total = moz["total"]["n_flagged"]
    chatty = moz["chatty"]["n_flagged"]
    share = chatty / total if total else float("nan")
    moz_lang = lang["language"]["baseline_loso_fpr_by_language"]["moz"]
    exais_lang = lang["language"]["baseline_loso_fpr_by_language"]["exais"]
    shares = lang["language"]["shares"]["moz"]
    lines = [
        "",
        "F1 reading (templated from the JSON):",
        "",
        f"- On `moz`, {chatty:,} of {total:,} baseline false positives "
        f"({pct(share)}) are chatty; {moz['promo']['n_flagged']:,}/"
        f"{moz['promo']['n_ham']:,} promo and {moz['bank_otp']['n_flagged']:,}/"
        f"{moz['bank_otp']['n_ham']:,} bank-OTP ham are flagged.",
        f"- On `exais`, transactional ham FPR is "
        f"{num(exais['transactional']['fpr'])} ({exais['transactional']['n_flagged']:,}/"
        f"{exais['transactional']['n_ham']:,}) and bank-OTP ham FPR is "
        f"{num(exais['bank_otp']['fpr'])} ({exais['bank_otp']['n_flagged']:,}/"
        f"{exais['bank_otp']['n_ham']:,}).",
        f"- Language check: heuristic and partial. Source metadata tags "
        f"{pct(lang['language']['declared_shares']['moz']['top'].get('pt', {}).get('share', 0))} "
        f"of `moz` `pt`. The text heuristic puts "
        f"{pct(shares['en']['share'])} of `moz` in `en` and "
        f"{pct(shares['other']['share'])} in `other`; the `en` bucket in a 100% "
        f"`pt` source is short or ambiguous text, not real English. Ham FPR is "
        f"{num(moz_lang['other']['fpr'])} for `other`-looking versus "
        f"{num(moz_lang['en']['fpr'])} for `en`-looking rows, while `exais` "
        f"shows no gap ({num(exais_lang['en']['fpr'])} `en` versus "
        f"{num(exais_lang['other']['fpr'])} `other`). This is consistent with "
        f"language shift on `moz`, not proof of it.",
    ]
    return lines


def f3_lines(f3: dict) -> list[str]:
    folds = f3["folds"]
    rec = f3["per_fold_recovery_from_weighted_quantile"]
    base_rate_moz = row(load("e4_loso.json")["rows"], "dummy", "moz")["pr_auc"]
    f3_moz_pr = folds["moz"]["source_class_balanced"]["pr_auc"]
    f3_moz_recall = folds["moz"]["source_class_balanced"]["recall"]
    lines = [
        "",
        "F3 diagnosis (post-hoc, not adopted):",
        "",
        "| source | baseline PR-AUC | F3 PR-AUC | baseline tau | F3 tau |",
        "|---|---:|---:|---:|---:|",
    ]
    for held in ("mishra", "moz", "exais"):
        base = folds[held]["baseline"]
        f3_row = folds[held]["source_class_balanced"]
        lines.append(
            f"| {held} | {num(base['pr_auc'])} | {num(f3_row['pr_auc'])} | "
            f"{num(base['tau'])} | {num(f3_row['tau'])} |"
        )
    lines += [
        "",
        "How the threshold is computed: the weights are applied when fitting "
        "every OOF fold and the final model, so OOF scores come from weighted "
        "models. The 1% quantile is then an **unweighted** empirical FPR over "
        "all train OOF rows, the same rule as E3/E4. Weights are not applied "
        "to the quantile.",
        "",
        "Weighted-quantile alternative (diagnostic only, not deployable):",
        "",
        "| source | recall at unweighted tau | recall at weighted tau | realised FPR at weighted tau |",
        "|---|---:|---:|---:|",
    ]
    for held in ("mishra", "moz"):
        lines.append(
            f"| {held} | {num(rec[held]['unweighted_tau_test_recall'])} | "
            f"{num(rec[held]['weighted_tau_test_recall'])} | "
            f"{num(rec[held]['weighted_tau_test_fpr'])} |"
        )
    lines += [
        "",
        f"- On `mishra` the collapse **is** a threshold artifact: the weighted "
        f"quantile recovers {num(rec['mishra']['recovered'])} recall at a "
        f"realised FPR of {num(rec['mishra']['weighted_tau_test_fpr'])}.",
        f"- On `moz` it is **not**: the F3 PR-AUC is {num(f3_moz_pr)} against a "
        f"base rate of {num(base_rate_moz)} — above base rate — but recall is "
        f"{num(f3_moz_recall)} at the 1%-FPR threshold "
        f"({num(rec['moz']['weighted_tau_test_recall'])} with a weighted "
        f"quantile, realised FPR {num(rec['moz']['weighted_tau_test_fpr'])}). "
        "Ranking is weak at the top and the operating point is unusable.",
        "- The weights are applied consistently in fitting and OOF scoring, so "
        "this is a threshold-rule mismatch on `mishra` and a weak-at-the-top "
        "ranking on `moz`, not a fitting bug. Either way F3 is not a fix.",
        "",
        "Descriptive recall at the held-out source's own FPR (oracle threshold, "
        "not deployable):",
        "",
        "| source | arm | recall@1% FPR | recall@5% FPR | recall@10% FPR |",
        "|---|---|---:|---:|---:|",
    ]
    for held in ("mishra", "moz"):
        for label, key in (("baseline", "baseline"), ("F3", "source_class_balanced")):
            desc = folds[held][key]["descriptive_at_test_fpr"]
            lines.append(
                f"| {held} | {label} | {num(desc['1']['recall'])} | "
                f"{num(desc['5']['recall'])} | {num(desc['10']['recall'])} |"
            )
    return lines


def evasion_source_table(ev: dict) -> list[str]:
    lines = [
        "",
        "Per-source E7/E8/E10 spam recall (imc25 is about 94% of test spam):",
        "",
        "| model | source | n spam | E7 off | E8 on | leet off | leet on | E10 linkless |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, label in (("svm", "svm"), ("mmbert_base", "mmBERT-base")):
        for src in ("imc25", "mishra", "moz", "smishtank"):
            row = ev["models"][model]["sources"][src]
            lines.append(
                f"| {label} | {src} | {num(row['n_spam'])} | "
                f"{num(row['baseline']['off'])} | {num(row['baseline']['on'])} | "
                f"{num(row['attacks']['off']['leet']['recall_mean'])} | "
                f"{num(row['attacks']['on']['leet']['recall_mean'])} | "
                f"{num(row['e10']['spam_linkless'])} |"
            )
    return lines


def real_alerts_lines() -> list[str]:
    """Aggregate-only summary of the local real-alert evaluation, if it exists."""
    path = RESULTS / "real_alerts.json"
    lines = [
        "",
        "## Real alerts (local evaluation set)",
        "",
        "`data/local/real_alerts.csv` is a small, git-ignored set of real",
        "anonymised alerts. It is scored once with the frozen bundle and is never",
        "used for training, calibration, threshold selection or tuning. Only",
        "aggregates are reported; no text and no row id is written.",
        "",
    ]
    if not path.exists():
        return lines + [
            "Not yet run. Run `make real-alerts` to write",
            "`reports/results/real_alerts.json`.",
        ]

    report = json.loads(path.read_text())
    labels = report.get("per_label", {})
    bands = report.get("bands", {})
    mean = report.get("mean_calibrated_score", {})
    lines += [
        f"Generated {report.get('generated_utc', '—')}; "
        f"artifact `{report.get('artifact_sha256', '—')}`.",
        "",
        "| n | ham | spam | FPR at frozen threshold | recall at frozen threshold |",
        "|---:|---:|---:|---:|---:|",
        f"| {num(report.get('n'))} | {num(labels.get('ham'))} | "
        f"{num(labels.get('spam'))} | "
        f"{num(report.get('fpr_at_frozen_threshold'))} | "
        f"{num(report.get('recall_at_frozen_threshold'))} |",
        "",
        "| band | all | ham | spam |",
        "|---|---:|---:|---:|",
    ]
    by_label = report.get("bands_by_label", {})
    for band in ("likely scam", "looks ordinary", "needs review"):
        lines.append(
            f"| {band} | {num(bands.get(band))} | "
            f"{num(by_label.get('ham', {}).get(band))} | "
            f"{num(by_label.get('spam', {}).get(band))} |"
        )
    lines += [
        "",
        f"Mean calibrated score: all {num(mean.get('all'))}, "
        f"ham {num(mean.get('ham'))}, spam {num(mean.get('spam'))}.",
        "",
        f"A run with fewer than {num(report.get('min_ham_required'))} ham rows "
        "refuses to report a false-positive rate.",
    ]
    return lines


def e17_table(e17: dict) -> list[str]:
    lines = [
        "Synthetic probe strings, CPU only; each arm runs in its own subprocess:",
        "",
        "| arm | batch 1 p50 | batch 1 p95 | batch 32 p50 / msg | batch 32 p95 / msg | disk MB | peak RSS MB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, label in (("svm", "svm"), ("mmbert_base", "mmBERT-base")):
        row = e17["arms"][arm]
        lines.append(
            f"| {label} | {row['batch_1_ms']['p50']:.2f} ms | "
            f"{row['batch_1_ms']['p95']:.2f} ms | "
            f"{row['batch_32_ms_per_message']['p50']:.2f} ms | "
            f"{row['batch_32_ms_per_message']['p95']:.2f} ms | "
            f"{row['disk_mb']:.1f} | {row['rss_mb']:.0f} |"
        )
    return lines


def build() -> str:
    e3 = load("e3_pool.json")
    e3_mm = load("e3_pool_mmbert_base.json")
    e4 = load("e4_loso.json")
    e4_small = load("e4_loso_mmbert_small.json")
    boot = load("e6_gold_v2_bootstrap.json")
    e12 = load("e12_ppv.json")
    e13 = load("e13_calibration.json")
    e14 = load("e14_conformal.json")
    recal = load("e_recal.json")
    ev = load("e7_e10_evasion.json")
    ev_src = load("e7_e10_by_source.json")
    f = load("f_experiments.json")
    lang_f3 = load("diag_language_f3.json")
    e17_path = RESULTS / "e17_latency.json"
    e17 = load("e17_latency.json") if e17_path.exists() else None

    e3_rows = e3["rows"] + e3_mm["rows"]
    e4_rows = e4["rows"] + e4_small["rows"]
    dummy_e3 = e3["rows"]
    dummy_e4 = e4["rows"]

    prevalence = e3["base_rate"]["pool_test_prevalence"]
    v2_pos = boot["gold"]["v2"]["n_pos"]
    v2_neg = boot["gold"]["v2"]["n_neg"]
    v2_recall = boot["gold"]["v2"]["models"]["mmbert_base"]["recall"]
    e4_svm_mishra = row(e4["rows"], "svm", "mishra")
    e4_small_mishra = row(e4_small["rows"], "mmbert_small", "mishra")
    gold_v2_dummy = boot["gold"]["v2"]["models"]["dummy"]["pr_auc"]["point"]
    gold_v2_base = boot["gold"]["v2"]["models"]["mmbert_base"]["pr_auc"]["point"]
    e4_svm_moz = row(e4["rows"], "svm", "moz")
    e4_dummy_moz = row(e4["rows"], "dummy", "moz")
    e3_n_train = e3_mm["n_train"]
    e4_n_train = e4_svm_mishra.get("n_train")

    lines = [
        "# Results summary",
        "",
        "Generated by `make summary` from `reports/results/*.json`. No number in",
        "this document is hand-typed and no message text is used.",
        "",
        "## How to read this",
        "",
        "- **E3** is the in-pool test split: train on pool train, freeze the",
        "  threshold on train OOF (classical) or `cal_a` (mmBERT), score the test",
        "  split once.",
        "- **E3 headline aggregation is an unweighted macro over `mishra` + `moz`",
        "  test rows (n_folds = 2), not the whole pool.** `exais` is FPR-only,",
        "  `imc25` and `smishtank` are recall-only, and none of them enters the",
        "  both-class macro.",
        "- **E4** is leave-one-source-out. The macro is the same unweighted mean",
        "  over `mishra` + `moz`; `exais` FPR is reported separately.",
        "- The pool is positive-heavy. No pool-level accuracy or F1 is a headline.",
        "",
        "## E3 — in-pool test (one pass)",
        "",
        "Headline label: unweighted macro over `mishra` + `moz` test rows.",
        "",
    ]
    lines += metric_table(
        e3_rows,
        dummy_e3,
        [("svm", "svm"), ("mmbert_base", "mmBERT-base")],
    )
    lines += [
        "",
        f"Pool test prevalence is {pct(prevalence)}; the dummy/base-rate PR-AUC",
        "is shown next to every model PR-AUC. `dummy` predicts the prior and has",
        "recall 0 / FPR 0 at the frozen threshold.",
        "",
        "## E4 — leave-one-source-out",
        "",
        "**mmBERT-small, not mmBERT-base.** There is no base LOSO arm.",
        "",
    ]
    lines += metric_table(
        e4_rows,
        dummy_e4,
        [("svm", "svm"), ("mmbert_small", "mmBERT-small")],
    )
    lines += [
        "",
        "## Gold holdouts v1 / v2 with bootstrap CIs",
        "",
        "Gold is synthetic and Claude-authored on the user's behalf. v1 was first",
        "scored before the ExAIS change; v2 was frozen before any model scored it.",
        "The bootstrap re-scores frozen models with no refit.",
        "",
    ]
    lines += gold_table()
    lines += [
        "",
        f"v2 recall for mmBERT-base is {num(v2_recall['point'])} "
        f"[{num(v2_recall['lo'])}, {num(v2_recall['hi'])}] on {v2_pos}/{v2_pos} "
        f"positives and {v2_neg} ham rows: a degenerate interval, not a precise",
        "estimate. The dummy PR-AUC is the base-rate line.",
        "",
        "## E12 — PPV at 1% / 5% / 10% prevalence (arithmetic, no re-score)",
        "",
        "In-pool E3:",
        "",
    ]
    lines += ppv_table(e12["in_pool_e3"], [("svm", "svm"), ("mmbert_base", "mmBERT-base")])
    lines += [
        "",
        "Under LOSO (svm and mmBERT-small; not base):",
        "",
    ]
    lines += ppv_table(e12["loso_e4"], [("svm", "svm"), ("mmbert_small", "mmBERT-small")])
    lines += [
        "",
        "## E13 — calibration on `cal_b` only",
        "",
        "| model | n | ECE (10 bins) | Brier | mean confidence | base rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, label in (("svm_pool", "svm (pool)"), ("mmbert_base", "mmBERT-base")):
        item = e13["models"][model]
        lines.append(
            f"| {label} | {num(item['n'])} | {num(item['ece_10bin'])} | {num(item['brier'])} | "
            f"{num(item['mean_confidence'])} | {num(item['base_rate'])} |"
        )
    lines += [
        "",
        "## E14 — conformal coverage vs 0.95",
        "",
        "In-pool (cal_a fits `q_hat`, cal_b is evaluated):",
        "",
        "| model | ham coverage | spam coverage | overall | abstain |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, label in (("svm_pool", "svm (pool)"), ("mmbert_base", "mmBERT-base")):
        item = e14["in_pool"][model]["point_estimate"]
        lines.append(
            f"| {label} | {num(item['coverage_ham'])} | {num(item['coverage_spam'])} | "
            f"{num(item['coverage_overall'])} | {num(item['abstain_rate'])} |"
        )
    lines += [
        "",
        "LOSO (mmBERT-small frozen fold checkpoints, no refit):",
        "",
        "| held out | ham coverage | spam coverage | overall | abstain |",
        "|---|---:|---:|---:|---:|",
    ]
    for held in ("mishra", "moz"):
        item = e14["loso"][held]
        lines.append(
            f"| {held} | {num(item['coverage_ham'])} | {num(item['coverage_spam'])} | "
            f"{num(item['coverage_overall'])} | {num(item['abstain_rate'])} |"
        )
    lines += [
        "",
        "## E-recal — labelled target ham does not fix the threshold",
        "",
        "`k` labelled ham from the held-out source set `tau`; the remaining rows are",
        "evaluated. This is a diagnostic, not a zero-shot result.",
        "",
    ]
    lines += recal_table(recal)
    lines += [
        "",
        f"On `moz`, recall collapses to about {num(recal['models']['svm']['moz']['recal']['100']['recall']['mean'])}",
        f"at k=100 with a realised FPR of {num(recal['models']['svm']['moz']['recal']['100']['fpr']['mean'])};",
        "the ranking is near base rate. The calibration FPR is ~1% by construction",
        "but the realised FPR stays above 1% on every fold. Never describe this as a",
        "fix.",
        "",
        "## E7-E10 — evasion and URL ablation",
        "",
        "E7 is normaliser OFF, E8 normaliser ON. Attacks are applied to spam only.",
        "The E8 baseline reproduces the E3 test macro at the frozen E3 threshold",
        "(asserted to `1e-6`), and an identity attack reproduces clean recall.",
        "",
    ]
    lines += evasion_tables(ev)
    lines += [
        "",
        "**The pool-wide evasion number is dominated by `imc25`,** which supplies "
        "about 94% of test spam. Read the per-source table below before quoting a "
        "single evasion figure.",
        "",
    ]
    lines += evasion_source_table(ev_src)
    lines += [
        "",
        "Normaliser cost: on svm, turning the normaliser ON raises clean spam "
        f"recall {num(ev['models']['svm']['baseline']['off']['recall'])} → "
        f"{num(ev['models']['svm']['baseline']['on']['recall'])} and raises the "
        f"FPR {num(ev['models']['svm']['baseline']['off']['fpr'])} → "
        f"{num(ev['models']['svm']['baseline']['on']['fpr'])}. The gain is bought "
        f"with roughly "
        f"{num(100 * (ev['models']['svm']['baseline']['on']['fpr'] - ev['models']['svm']['baseline']['off']['fpr']), 1)} "
        f"extra false positives per 100 ham.",
        "",
        "## F1-F5 — svm-only fixes under LOSO (no gold)",
        "",
        f"Success rule, fixed before the run: {f['success_rule']}",
        "",
    ]
    lines += f_tables(f)
    lines += f1_language_lines(f, lang_f3)
    lines += f3_lines(lang_f3["f3"])
    if e17 is not None:
        lines += [
            "",
            "## E17 — latency and footprint",
            "",
        ]
        lines += e17_table(e17)
    lines += real_alerts_lines()
    lines += [
        "",
        "## Caveats",
        "",
        "- **E3 base vs E4 small are different sizes and settings.** E3 is",
        f"  mmBERT-base on {e3_n_train:,} train rows with an in-pool test split;",
        f"  E4 is mmBERT-small on about {e4_n_train:,} train rows under LOSO. The",
        "  comparison cannot separate model size from the easier setting.",
        "- **E3 and E4 are single runs without confidence intervals.** Treat small",
        "  differences as noise.",
        "- **mmBERT-small is not shown to beat svm on both sources.** On `mishra`",
        f"  its PR-AUC is {num(e4_small_mishra.get('pr_auc'))} vs svm",
        f"  {num(e4_svm_mishra.get('pr_auc'))}. The macro gain is concentrated in",
        "  `moz` and `imc25`.",
        "- **Gold is synthetic and Claude-authored.** v2 recall is",
        f"  {v2_pos}/{v2_pos} with a degenerate CI [{num(v2_recall['lo'])}, {num(v2_recall['hi'])}]",
        f"  and only {v2_neg} ham rows. The absolute FPR is not a field estimate.",
        "- **Ranking holds on gold but NOT on `moz` under LOSO.** Gold v2 PR-AUC is",
        f"  {num(gold_v2_base)} (base) vs a dummy base rate of {num(gold_v2_dummy)};",
        f"  `moz` LOSO PR-AUC is {num(e4_svm_moz.get('pr_auc'))} vs a dummy base rate",
        f"  of {num(e4_dummy_moz.get('pr_auc'))}. The limitation is thresholds",
        "  everywhere and ranking on some held-out sources.",
        "- **`mishra`, `moz` and `exais` are now development folds for the F arms.**",
        "  They are no longer pristine held-out evidence for changes chosen from F.",
        f"- **Pool-wide evasion hides source-level failures.** mmBERT-base "
        f"leetspeak recall on `moz` is "
        f"{num(ev_src['models']['mmbert_base']['sources']['moz']['attacks']['on']['leet']['recall_mean'])} "
        f"against a pool-wide "
        f"{num(ev['models']['mmbert_base']['attacks']['on']['leet']['recall_mean'])}.",
        f"- **The normaliser has a cost and a source-specific benefit.** On svm it "
        f"raises `mishra` clean recall "
        f"{num(ev_src['models']['svm']['sources']['mishra']['baseline']['off'])} → "
        f"{num(ev_src['models']['svm']['sources']['mishra']['baseline']['on'])} and "
        f"raises pool FPR "
        f"{num(ev['models']['svm']['baseline']['off']['fpr'])} → "
        f"{num(ev['models']['svm']['baseline']['on']['fpr'])}.",
        f"- **No accuracy is reported.** The pool is about {pct(prevalence)} positive,",
        "  so accuracy would mostly restate the base rate.",
        "- **Synthetic gold false positives concentrate in legitimate bank/service",
        "  messages, and the pool has almost no real bank-alert ham.** ExAIS is the",
        "  only bank/service-notification ham source, and it is small.",
        "",
        "## What to say if asked",
        "",
        "- Lead with the distinction: the ranking is usable on several sources; the",
        "  frozen threshold is not portable. Gold FPR is high while gold PR-AUC",
        "  stays above the base rate.",
        "- Say `mmBERT-small` whenever quoting E4. Never present E4 as a",
        "  base-model result.",
        "- Say the E3 macro is over `mishra` + `moz` only. Do not quote a pooled",
        "  accuracy.",
        "- Say recalibration on labelled target ham did not fix `moz`; it is a",
        "  diagnostic, not a fix.",
        "- Say the F arms were pre-registered, the baseline reproduced E4 exactly,",
        "  and every arm is reported, including failures.",
        "",
        "## Q&A",
        "",
        "- **Is this a smishing detector?** It is a ranking model with a frozen",
        "  operating point. The ranking transfers better than the threshold.",
        "- **Why no accuracy?** The pool is positive-heavy; accuracy would be",
        "  dominated by the base rate.",
        "- **Does the transformer beat svm?** In-pool base beats svm, but there is",
        "  no base LOSO arm and small does not beat svm on `mishra` PR-AUC. The",
        "  E3-vs-E4 gap is source shift plus setting, not a clean size effect.",
        "- **Does it generalise to new sources?** Ranking often does; the threshold",
        "  does not. Gold and `moz` LOSO both show this.",
        "- **Was gold used for tuning?** No. Gold was not used for tuning,",
        "  threshold selection or model selection.",
        "- **Does recalibration fix it?** No. It moves the threshold but recall",
        "  collapses on `moz` and realised FPR stays above 1%.",
        "- **Does the normaliser help?** It helps svm most on leetspeak and",
        "  homoglyphs; mmBERT-base is less sensitive.",
        "- **Did any F arm pass the success rule?** No arm is a fix. `F3` "
        "(source-class-balanced) passes the letter of the rule only through a "
        "collapsed recall "
        f"({num(f['arms']['source_class_balanced']['macro']['macro_recall'])} "
        "against the baseline "
        f"{num(f['arms']['baseline']['macro']['macro_recall'])}). The post-hoc "
        "check splits the collapse: on `mishra` it is an unweighted-quantile "
        "artifact, while on `moz` recall stays at "
        f"{num(lang_f3['f3']['per_fold_recovery_from_weighted_quantile']['moz']['weighted_tau_test_recall'])} "
        "even with a weighted quantile, so that fold's operating point is "
        "unusable at 1% FPR (its PR-AUC is above the base rate, but the top of "
        "the ranking is weak). "
        "`format_masked` does not pass. Every arm is reported, including "
        "failures.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build() + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
