"""Post-hoc diagnostics: language shares, F3 threshold audit, per-source evasion.

These are exploratory. No arm is adopted, no threshold is changed, and no test
row is used to fit anything. Outputs are aggregates only; the per-row score
dump keeps hashed ids, labels and scores, never text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from smishlens.evaluate import metrics_at, threshold_for_fpr
from smishlens.f_experiments import (
    BOTH,
    FOLDS,
    _run_arm,
    _source_class_weights,
)
from smishlens.pool_experiments import RESULTS, ROOT, _fold, load_pool

LOCAL = ROOT / "artifacts" / "diagnostics"
FPR_TARGETS = (0.01, 0.05, 0.10)

# Coarse English and Hinglish marker lists. This is a stopword heuristic, NOT a
# language-ID model. It only counts token matches; no message is printed.
EN = set(
    "the and you your for with this that have from will are can not but all our "
    "out get just now how what when who why was were been has had his her its "
    "they them their there here about into over after before more some any "
    "please call text message send today tomorrow tonight ok okay yes no".split()
)
HI = set(
    "hai hain nahi nahin aap aapka aapki aapke kya kyun kaise kar karo kare "
    "karna karta karti mera meri mujhe tum tumhara bhai paise paisa jaldi abhi "
    "aaj kal accha theek bhej bhejo milega milegi chahiye hoga hogi kuch bahut "
    "kaam karo please".split()
)


def guess_language(text: object) -> str:
    tokens = re.findall(r"[a-z]+", str(text).lower())
    if not tokens:
        return "other"
    en = sum(t in EN for t in tokens) / len(tokens)
    hi = sum(t in HI for t in tokens) / len(tokens)
    if hi >= 0.04 and hi >= en:
        return "hinglish"
    if en >= 0.04:
        return "en"
    return "other"


def _share(values: pd.Series) -> dict:
    counts = values.value_counts().to_dict()
    n = int(len(values))
    return {
        "n": n,
        **{k: {"n": int(counts.get(k, 0)), "share": counts.get(k, 0) / n} for k in ("en", "hinglish", "other")},
    }


def _declared_share(values: pd.Series) -> dict:
    """Source-declared language tags, normalised. Metadata, not text inference."""
    declared = (
        values.astype(str)
        .str.lower()
        .str.strip()
        .replace({"nan": "unlabelled", "english": "en", "portuguese": "pt"})
    )
    counts = declared.value_counts().to_dict()
    n = int(len(declared))
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
    return {
        "n": n,
        "top": {k: {"n": int(v), "share": v / n} for k, v in top},
    }


def language_block(frame: pd.DataFrame, raw: dict) -> dict:
    labelled = frame.assign(language=frame["text"].map(guess_language))
    shares = {
        str(src): _share(sub["language"])
        for src, sub in labelled.groupby("source", sort=True)
    }
    declared = {
        str(src): _declared_share(sub["lang"])
        for src, sub in frame.groupby("source", sort=True)
    }
    by_lang: dict[str, dict] = {}
    for held in ("mishra", "moz", "exais"):
        test = raw[held]["test"].copy()
        test["language"] = test["text"].map(guess_language)
        p = np.asarray(raw[held]["p"], dtype=float)
        tau = raw[held]["tau"]
        y = test["y"].to_numpy()
        rows = {}
        for lang, sub in test[y == 0].assign(score=p[y == 0]).groupby("language", sort=True):
            n = int(len(sub))
            flagged = int((sub["score"] >= tau).sum())
            rows[str(lang)] = {
                "n_ham": n,
                "n_flagged": flagged,
                "fpr": flagged / n if n else None,
            }
        by_lang[held] = rows
    return {
        "method": (
            "stopword heuristic (English function words and Hinglish markers), "
            "not a language-ID model; counts only. The `other` bucket also "
            "catches very short messages with no function words, so it is not "
            "a pure non-English flag."
        ),
        "declared_shares": declared,
        "shares": shares,
        "baseline_loso_fpr_by_language": by_lang,
    }


def _weighted_threshold(scores, y, weights, max_fpr: float = 0.01) -> dict:
    """Threshold on the weighted OOF FPR, for the F3 audit only."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    s, yy, ww = scores[order], y[order], w[order]
    neg = ww[yy == 0].sum()
    pos = ww[yy == 1].sum()
    fpr = np.cumsum(ww * (yy == 0)) / neg
    tpr = np.cumsum(ww * (yy == 1)) / pos
    within = np.where(fpr <= max_fpr)[0]
    i = int(within[-1]) if len(within) else 0
    return {"tau": float(s[i]), "weighted_oof_tpr": float(tpr[i]), "weighted_oof_fpr": float(fpr[i])}


def _descriptive_at_test_fpr(y, p, target: float) -> dict:
    """Oracle threshold on the held-out rows: descriptive, not deployable."""
    tau, tpr, fpr = threshold_for_fpr(y, p, target)
    return {"target_fpr": target, "tau": float(tau), "recall": float(tpr), "realised_fpr": float(fpr)}


def _fold_block(held: str, raw: dict, weights: np.ndarray | None) -> dict:
    test = raw[held]["test"]
    y = test["y"].to_numpy()
    p = np.asarray(raw[held]["p"], dtype=float)
    m = metrics_at(y, p, raw[held]["tau"])
    row = {
        "mode": "fpr_only" if held == "exais" else "both",
        "tau": raw[held]["tau"],
        "n_pos": m["n_pos"],
        "n_neg": m["n_neg"],
        "recall": m["recall"] if held != "exais" else None,
        "fpr": m["fpr"],
        "pr_auc": m["pr_auc"] if held != "exais" else None,
        "descriptive_at_test_fpr": {
            str(int(t * 100)): _descriptive_at_test_fpr(y, p, t) for t in FPR_TARGETS
        },
    }
    if weights is not None:
        alt = _weighted_threshold(raw[held]["oof"], raw[held]["train_y"], weights)
        alt_m = metrics_at(y, p, alt["tau"])
        row["weighted_oof_threshold"] = {
            **alt,
            "test_recall": alt_m["recall"] if held != "exais" else None,
            "test_fpr": alt_m["fpr"],
        }
    return row


def _row_hash(source: str, text: str) -> str:
    return hashlib.sha256(f"{source}\x00{text}".encode()).hexdigest()[:16]


def _dump_scores(frame: pd.DataFrame, runs: dict) -> Path:
    LOCAL.mkdir(parents=True, exist_ok=True)
    path = LOCAL / "loso_scores.csv"
    rows = []
    for model, (_, raw) in runs.items():
        for held, block in raw.items():
            test = block["test"]
            for (_, r), score in zip(test.iterrows(), block["p"], strict=True):
                rows.append(
                    {
                        "model": model,
                        "held_out": held,
                        "row_hash": _row_hash(str(r["source"]), str(r["text"])),
                        "y": int(r["y"]),
                        "score": float(score),
                        "tau": float(block["tau"]),
                        "language": guess_language(r["text"]),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def diagnose_f3(
    frame: pd.DataFrame, baseline: dict, baseline_raw: dict, f3: dict, f3_raw: dict
) -> dict:
    folds = {}
    for held in FOLDS:
        train, _ = _fold(frame, held)
        w = _source_class_weights(train)
        folds[held] = {
            "baseline": _fold_block(held, baseline_raw, None),
            "source_class_balanced": _fold_block(held, f3_raw, w),
        }

    base_macro = baseline["macro"]
    f3_macro = f3["macro"]
    recovery = {}
    for held in BOTH:
        f3_row = folds[held]["source_class_balanced"]
        recovery[held] = {
            "unweighted_tau_test_recall": f3_row["recall"],
            "weighted_tau_test_recall": f3_row["weighted_oof_threshold"]["test_recall"],
            "recovered": (
                f3_row["weighted_oof_threshold"]["test_recall"] - f3_row["recall"]
            ),
            "weighted_tau_test_fpr": f3_row["weighted_oof_threshold"]["test_fpr"],
        }
    alt_recalls = [recovery[h]["weighted_tau_test_recall"] for h in BOTH]
    alt_fprs = [recovery[h]["weighted_tau_test_fpr"] for h in BOTH]
    return {
        "note": (
            "post-hoc exploratory; no arm is adopted. Source-class weights are "
            "applied when fitting every OOF fold and the final model. The 1% "
            "quantile is an UNWEIGHTED empirical FPR over all train OOF rows, "
            "which is the same rule used by E3/E4."
        ),
        "weights": {
            "applied_in_fit": True,
            "applied_in_oof_scores": True,
            "applied_in_threshold_quantile": False,
            "threshold_rule": "max recall s.t. unweighted empirical FPR <= 0.01 over all train OOF rows",
            "weight_definition": "1 / rows in the (source, class) cell",
        },
        "folds": folds,
        "macro": {
            "baseline": base_macro,
            "source_class_balanced": f3_macro,
            "delta_fpr": f3_macro["macro_fpr"] - base_macro["macro_fpr"],
            "delta_pr_auc": f3_macro["macro_pr_auc"] - base_macro["macro_pr_auc"],
        },
        "per_fold_recovery_from_weighted_quantile": recovery,
        "weighted_quantile_alternative": {
            "macro_recall": float(np.mean(alt_recalls)),
            "macro_fpr": float(np.mean(alt_fprs)),
            "note": "diagnostic only; it weights the quantile, not a deployable rule",
        },
        "verdict": (
            "Both, and the split is per fold. On `mishra` the collapse is a "
            "threshold artifact: an unweighted OOF quantile applied to scores "
            f"from a weighted model; a weighted quantile recovers "
            f"{recovery['mishra']['recovered']:.4f} recall there. On `moz` it is "
            "NOT a threshold artifact: recall stays "
            f"{recovery['moz']['weighted_tau_test_recall']:.4f} at a realised FPR "
            f"of {recovery['moz']['weighted_tau_test_fpr']:.4f}, so that fold is a "
            "score-scale/ranking failure. The weights are applied consistently in "
            "fitting and OOF scoring; only the quantile is unweighted, the same "
            "rule as E3/E4. Either way F3 is not a fix."
        ),
        "scores_file": "artifacts/diagnostics/loso_scores.csv (local, untracked)",
    }


def run() -> dict:
    frame = load_pool()
    baseline, baseline_raw = _run_arm(frame)
    f3, f3_raw = _run_arm(frame, weights_fn=_source_class_weights, class_weight=None)
    f3_block = diagnose_f3(frame, baseline, baseline_raw, f3, f3_raw)
    language = language_block(frame, baseline_raw)
    _dump_scores(
        frame,
        {
            "baseline": (baseline, baseline_raw),
            "source_class_balanced": (f3, f3_raw),
        },
    )
    return {"language": language, "f3": f3_block}


def _finite(obj):
    """NaN/inf are not valid JSON: emit null instead."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "diag_language_f3.json"))
    args = ap.parse_args(argv)
    payload = _finite(run())
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
