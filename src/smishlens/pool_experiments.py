"""Tier-2 experiment runner: E5, E3, E4, E6 (+ gold once), classical + mmBERT.

Rules baked in (roadmap §8, and the 0.923 base rate):

- Per-source rows plus a **macro average over both-class sources only**. Never a
  pooled micro-average, never accuracy, never a pooled F1 as a headline.
- ``mishra`` and ``moz`` are the only both-class sources; ``imc25`` and
  ``smishtank`` report recall only, even though three IMC'25 rows are ham after
  the label audit.
- Thresholds (FPR <= 1% and the 10:1 cost point) are frozen on training
  out-of-fold scores only. The pool test split is scored once per model.
- Model selection uses train OOF macro PR-AUC over the both-class sources only.
- Gold is synthetic and is scored once, after selection, never for tuning.

Usage
-----
    python -m smishlens.pool_experiments --only E5
    python -m smishlens.pool_experiments --only E3,E4,E6
    python -m smishlens.pool_experiments --only E5,E3,E4,E6 --models dummy,mnb,cnb,lr,svm
    python -m smishlens.pool_experiments --only "" --mmbert-e3 base
    python -m smishlens.pool_experiments --only "" --mmbert-e4 small
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from smishlens.evaluate import cost_operating_point, metrics_at, ppv, threshold_for_fpr
from smishlens.features import build_features
from smishlens.local_data import require_gold, require_uci
from smishlens.pool import BOTH_CLASSES, HAM_ONLY_SOURCES
from smishlens.probes import _cv_auc, format_features, shortcut_report

ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = ROOT / "data" / "processed" / "modern"
UCI_DIR = ROOT / "data" / "processed" / "uci"
GOLD = ROOT / "data" / "gold_holdout.csv"
RESULTS = ROOT / "reports" / "results"
LOG = ROOT / "reports" / "verification_log.md"

SEED = 42
MAX_FPR = 0.01
PREVALENCES = (0.05, 0.10, 0.20)
RECALL_ONLY = ("imc25", "smishtank")
FPR_ONLY = HAM_ONLY_SOURCES
FPR_SOURCES = BOTH_CLASSES
HELD_OUT = (*BOTH_CLASSES, *RECALL_ONLY, *FPR_ONLY)
DEFAULT_MODELS = ("dummy", "mnb", "cnb", "lr", "svm")
MMBERT_MODELS = {
    "base": str(ROOT / "models" / "mmBERT-base"),
    "small": str(ROOT / "models" / "mmBERT-small"),
}

NOTE = (
    "pool prevalence is positive-heavy; IMC'25 dominates the rows and has only "
    "3 ham, so no pool-level accuracy or F1 is a headline; FPR is reported on "
    "the both-class sources (mishra, moz) and separately on the exais ham-only "
    "fold; imc25 and smishtank report recall only; exais never enters a "
    "both-class macro"
)


def _est(name: str):
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "mnb":
        return MultinomialNB(alpha=0.1)
    if name == "cnb":
        return ComplementNB(alpha=0.1)
    if name == "lr":
        return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED)
    if name == "svm":
        return LinearSVC(C=0.5, class_weight="balanced", dual="auto", random_state=SEED)
    raise ValueError(f"unknown model {name!r}")


def pipe_for(name: str) -> Pipeline:
    """Same feature block for every model so the comparison varies one thing."""
    return Pipeline([("feats", build_features()), ("clf", _est(name))])


def load_pool() -> pd.DataFrame:
    parts = [pd.read_csv(POOL_DIR / f"{s}.csv") for s in ("train", "cal", "test")]
    for name, part in zip(("train", "cal", "test"), parts, strict=True):
        part["split"] = name
    return pd.concat(parts, ignore_index=True)


def load_uci_train() -> pd.DataFrame:
    require_uci()
    return pd.read_csv(UCI_DIR / "train.csv")


def load_gold() -> pd.DataFrame:
    require_gold()
    gold = pd.read_csv(GOLD, dtype=str, keep_default_na=False)
    gold["y3"] = gold["label"].str.strip().str.lower()
    gold["y"] = (gold["y3"] != "ham").astype(int)
    return gold


def source_weights(train: pd.DataFrame) -> np.ndarray:
    """Weight every source equally, so IMC'25's 15,657 rows cannot dominate.

    Weighting by source and class was rejected: IMC'25 has three ham rows in
    train, so that cell would carry a 754x per-row weight and the arm would
    measure three rows. Weighting by source only keeps the knob honest.
    """
    counts = train.groupby("source")["text"].transform("size")
    w = 1.0 / counts.to_numpy(dtype=float)
    return w / w.mean()


def mode_for(source: str) -> str:
    """One class only where the source only has one class.

    ``exais`` is ham-only by construction: its positives are dropped before
    dedup, so it can only report a false-positive rate and must never enter a
    both-class macro.
    """
    if source in RECALL_ONLY:
        return "recall_only"
    if source in FPR_ONLY:
        return "fpr_only"
    return "both"


def _decision(pipe: Pipeline, texts) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(texts)[:, 1]
    return pipe.decision_function(texts)


def _method(pipe: Pipeline) -> str:
    return "predict_proba" if hasattr(pipe, "predict_proba") else "decision_function"


def oof_scores(
    train: pd.DataFrame, name: str, weights: np.ndarray | None = None, n_jobs: int = 1
) -> np.ndarray:
    pipe = pipe_for(name)
    params = {} if weights is None else {"clf__sample_weight": weights}
    oof = cross_val_predict(
        pipe,
        train["text"],
        train["y"],
        method=_method(pipe),
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        params=params,
        n_jobs=n_jobs,
    )
    return oof[:, 1] if oof.ndim == 2 else oof


def oof_macro_pr_auc(train: pd.DataFrame, oof: np.ndarray) -> float:
    """Macro PR-AUC over both-class sources, on train OOF scores only."""
    values = []
    for src in BOTH_CLASSES:
        mask = (train["source"] == src).to_numpy()
        y = train.loc[mask, "y"].to_numpy()
        if y.size and len(np.unique(y)) == 2:
            values.append(float(average_precision_score(y, oof[mask])))
    return float(np.mean(values)) if values else float("nan")


def _scores(y, p, tau: float, mode: str) -> dict:
    """Metrics a fold's mode supports; never a metric the mode cannot estimate.

    ``both`` reports recall, FPR, both AUCs and prevalence-adjusted PPV.
    ``recall_only`` reports recall only. ``fpr_only`` (a ham-only source such as
    ExAIS) reports FPR only. Keeping the keys disjoint is what stops a one-class
    fold from being averaged into a both-class macro.
    """
    m = metrics_at(y, p, tau)
    out = {"n_pos": m["n_pos"], "n_neg": m["n_neg"]}
    if mode == "both":
        out["recall"] = m["recall"]
        out["fpr"] = m["fpr"]
        out["pr_auc"] = m["pr_auc"]
        out["roc_auc"] = m["roc_auc"]
        for prev in PREVALENCES:
            out[f"ppv_{int(prev * 100)}pct"] = ppv(m["recall"], m["fpr"], prev)
    elif mode == "recall_only":
        out["recall"] = m["recall"]
    elif mode == "fpr_only":
        out["fpr"] = m["fpr"]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return out


def evaluate(train, test, name, weights=None) -> dict:
    """Fit once, freeze both thresholds on train OOF, score test once."""
    oof = oof_scores(train, name, weights)
    tau = float(threshold_for_fpr(train["y"].to_numpy(), oof, MAX_FPR)[0])
    cost = cost_operating_point(train["y"].to_numpy(), oof, fp_cost=10.0, fn_cost=1.0)
    params = {} if weights is None else {"clf__sample_weight": weights}
    pipe = pipe_for(name).fit(train["text"], train["y"], **params)
    p = _decision(pipe, test["text"])
    return {
        "tau": tau,
        "cost_tau": float(cost["tau"]),
        "oof_macro_pr_auc": oof_macro_pr_auc(train, oof),
        "p": p,
        "pipe": pipe,
    }


def _row(base: dict, mode: str, y, p, tau, cost_tau) -> dict:
    row = {**base, "mode": mode, "tau": tau, "cost_tau": cost_tau}
    row.update(_scores(y, p, tau, mode))
    cost = _scores(y, p, cost_tau, mode)
    if "recall" in cost:
        row["recall_cost"] = cost["recall"]
    if "fpr" in cost:
        row["fpr_cost"] = cost["fpr"]
    return row


def _macro(rows: list[dict]) -> dict:
    """Unweighted mean over both-class rows only, with the fold count beside it.

    ``fpr_only`` and ``recall_only`` rows are excluded here, not merely absent
    from the metric keys: a source that cannot estimate recall must not silently
    contribute a zero to a recall average.
    """
    both = [r for r in rows if r["mode"] == "both"]
    out = {"n_folds": len(both)}
    for key in ("recall", "fpr", "pr_auc", "roc_auc", "ppv_5pct", "ppv_10pct", "ppv_20pct"):
        values = [r[key] for r in both if key in r and r[key] == r[key]]
        if values:
            out[f"macro_{key}"] = float(np.mean(values))
    return out


# --- cached matrices (E6 onwards) -----------------------------------------


class FoldMatrices:
    """One fold's features, vectorised once and shared by every model.

    The vectoriser is fitted on that fold's TRAIN rows only, so the held-out
    source contributes no vocabulary. The five CV splits used to estimate the
    threshold then index into the cached matrix instead of re-vectorising.
    Measured cost of that choice versus refitting inside each CV split: same
    vocabulary, bitwise-identical test scores, threshold shifts by ~0.02 on
    decision-function scores. See the log entry.
    """

    def __init__(self, feats, X, y, source, splits, X_test=None):
        self.feats = feats
        self.X = X
        self.y = y
        self.source = source
        self.splits = splits
        self.X_test = X_test


def build_fold_matrices(train: pd.DataFrame, test: pd.DataFrame | None = None) -> FoldMatrices:
    feats = build_features()
    X = feats.fit_transform(train["text"])
    y = train["y"].to_numpy()
    splits = list(
        StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)
    )
    X_test = feats.transform(test["text"]) if test is not None else None
    return FoldMatrices(feats, X, y, train["source"].to_numpy(), splits, X_test)


def transform_with(mats: FoldMatrices, texts):
    return mats.feats.transform(texts)


def _score_est(est, X) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def oof_cached(mats: FoldMatrices, name: str, weights: np.ndarray | None = None) -> np.ndarray:
    oof = np.empty(len(mats.y), dtype=float)
    for tr, va in mats.splits:
        est = _est(name)
        kw = {} if weights is None else {"sample_weight": weights[tr]}
        est.fit(mats.X[tr], mats.y[tr], **kw)
        oof[va] = _score_est(est, mats.X[va])
    return oof


def evaluate_cached(mats: FoldMatrices, name: str, weights: np.ndarray | None = None) -> dict:
    """Same contract as ``evaluate`` but on pre-vectorised matrices."""
    oof = oof_cached(mats, name, weights)
    tau = float(threshold_for_fpr(mats.y, oof, MAX_FPR)[0])
    cost = cost_operating_point(mats.y, oof, fp_cost=10.0, fn_cost=1.0)
    est = _est(name)
    kw = {} if weights is None else {"sample_weight": weights}
    est.fit(mats.X, mats.y, **kw)
    p = _score_est(est, mats.X_test)
    return {
        "tau": tau,
        "cost_tau": float(cost["tau"]),
        "oof_macro_pr_auc": _macro_pr_auc(mats, oof),
        "p": p,
        "est": est,
    }


def _macro_pr_auc(mats: FoldMatrices, oof: np.ndarray) -> float:
    values = []
    for src in BOTH_CLASSES:
        mask = mats.source == src
        y = mats.y[mask]
        if y.size and len(np.unique(y)) == 2:
            values.append(float(average_precision_score(y, oof[mask])))
    return float(np.mean(values)) if values else float("nan")


# --- E5 -------------------------------------------------------------------


def run_e5(frame: pd.DataFrame, min_per_class: int = 20) -> dict:
    report = shortcut_report(frame, min_per_class=min_per_class, fpr_only=FPR_ONLY)
    feats = format_features(frame["text"])
    # Keep the IMC'25-vs-rest probe on the same row set as the pooled class
    # AUC, so the pre-ExAIS number stays comparable. ExAIS gets its own probe.
    class_rows = ~frame["source"].isin(FPR_ONLY).to_numpy()
    is_imc = ((frame["source"] == "imc25") & class_rows).to_numpy()
    imc_auc = _cv_auc(feats.loc[class_rows].reset_index(drop=True), is_imc[class_rows].astype(int))
    is_exais = (frame["source"].isin(FPR_ONLY)).to_numpy()
    exais_auc = _cv_auc(feats, is_exais.astype(int)) if is_exais.any() else float("nan")
    return {
        "note": NOTE,
        "scope": "whole frozen pool (train+cal+test); descriptive probe of the data, no model is selected and no threshold is set here",
        "n_rows": int(len(frame)),
        "format_only_auc_pooled": report["format_only_auc"],
        "length_only_auc_pooled": report["length_only_auc"],
        "class_auc_scope": report["class_auc_scope"],
        "by_source": report["by_source"],
        "fpr_only": report["fpr_only"],
        "source_id_acc_from_format": report["source_id_acc_from_format"],
        "source_id_chance": report["source_id_chance"],
        "source_id_sources": report["source_id_sources"],
        "imc25_vs_rest_auc_from_format": imc_auc,
        "exais_vs_rest_auc_from_format": exais_auc,
        "imc25_row_share": float((frame["source"] == "imc25").mean()),
        "exais_row_share": float(is_exais.mean()),
    }


# --- selection (train only) -----------------------------------------------


SELECTION_RULE = (
    "Highest train-split out-of-fold macro PR-AUC over the two both-class sources "
    "(mishra, moz). Tie-break, on those same two sources only: lower train-OOF FPR "
    "at the OOF-fixed 1% threshold, then higher train-OOF recall. Test-split and "
    "LOSO numbers are not inputs; fpr_only sources are never inputs."
)


def run_selection(frame: pd.DataFrame, models: list[str]) -> dict:
    """Choose the headline from train-split OOF evidence only.

    Same estimator, same five stratified folds and same OOF-fixed threshold as
    E3, so the pick matches ``e3_pool.json``. No test row and no LOSO number is
    read here.
    """
    train = frame[frame["split"] == "train"]
    y = train["y"].to_numpy()
    # Selection and its tie-break read the two both-class sources only. The
    # threshold itself is frozen on all train rows, because ExAIS ham is real
    # negative evidence for the operating point; it just cannot vote on which
    # model wins.
    both_mask = train["source"].isin(BOTH_CLASSES).to_numpy()
    rows = []
    for name in models:
        oof = oof_scores(train, name)
        tau, _, _ = threshold_for_fpr(y, oof, MAX_FPR)
        cost = cost_operating_point(y, oof, fp_cost=10.0, fn_cost=1.0)
        overall = metrics_at(y[both_mask], oof[both_mask], tau)
        per_source = {}
        for src in BOTH_CLASSES:
            mask = (train["source"] == src).to_numpy()
            m = metrics_at(y[mask], oof[mask], tau)
            per_source[src] = {
                "n_pos": m["n_pos"],
                "n_neg": m["n_neg"],
                "recall": m["recall"],
                "fpr": m["fpr"],
                "pr_auc": m["pr_auc"],
            }
        rows.append(
            {
                "model": name,
                "oof_macro_pr_auc": oof_macro_pr_auc(train, oof),
                "oof_tau": tau,
                "oof_fpr": overall["fpr"],
                "oof_recall": overall["recall"],
                "oof_cost_tau": float(cost["tau"]),
                "oof_by_source": per_source,
            }
        )

    ranked = sorted(rows, key=lambda r: (-r["oof_macro_pr_auc"], r["oof_fpr"], -r["oof_recall"]))
    return {
        "note": NOTE,
        "rule": SELECTION_RULE,
        "evidence": "train split only, 5-fold OOF at seed 42; test split and LOSO folds untouched",
        "n_train": int(len(train)),
        "n_train_both_classes": int(both_mask.sum()),
        "fpr_only_sources_excluded_from_selection": list(FPR_ONLY),
        "ranked": ranked,
        "selected_model": ranked[0]["model"],
        "not_inputs": ["pool test split", "LOSO folds", "gold holdout"],
    }


# --- E3 -------------------------------------------------------------------


def run_e3(frame: pd.DataFrame, models: list[str]) -> tuple[dict, str]:
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    rows: list[dict] = []
    selection: list[tuple[float, str]] = []
    for name in models:
        res = evaluate(train, test, name)
        selection.append((res["oof_macro_pr_auc"], name))
        block: list[dict] = []
        for src in sorted(test["source"].unique()):
            mask = (test["source"] == src).to_numpy()
            block.append(
                _row(
                    {"experiment": "E3", "model": name, "source": src},
                    mode_for(src),
                    test.loc[mask, "y"].to_numpy(),
                    res["p"][mask],
                    res["tau"],
                    res["cost_tau"],
                )
            )
        rows.extend(block)
        rows.append(
            {
                "experiment": "E3",
                "model": name,
                "source": "macro",
                "mode": "macro",
                **_macro(block),
            }
        )
    selection.sort(reverse=True)
    best = selection[0][1]
    prevalence = float(test["y"].mean())
    payload = {
        "note": NOTE,
        "selection_rule": "max train-OOF macro PR-AUC over moz and mishra; fpr_only sources never enter the macro",
        "selection": [{"model": m, "oof_macro_pr_auc": v} for v, m in selection],
        "selected_model": best,
        "sources": sorted(test["source"].unique().tolist()),
        "fpr_only_sources": list(FPR_ONLY),
        # "Always spam" is not a result. It is here so the base rate is visible
        # next to every other row and cannot be mistaken for skill.
        "base_rate": {
            "pool_test_rows": int(len(test)),
            "pool_test_prevalence": prevalence,
            "always_spam": {
                "recall": 1.0,
                "fpr": 1.0,
                "precision": prevalence,
                "accuracy": prevalence,
            },
            "why": f"shown only to make the {prevalence:.3f} pool base rate visible; never a headline",
        },
        "rows": rows,
    }
    return payload, best


# --- E4 -------------------------------------------------------------------


def _fold(frame: pd.DataFrame, held: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[(frame["split"] == "train") & (frame["source"] != held)]
    test = frame[frame["source"] == held]
    return train, test


def run_e4(frame: pd.DataFrame, models: list[str]) -> dict:
    rows: list[dict] = []
    for name in models:
        start = len(rows)
        for held in HELD_OUT:
            train, test = _fold(frame, held)
            res = evaluate(train, test, name)
            rows.append(
                _row(
                    {"experiment": "E4", "model": name, "source": held, "n_train": len(train)},
                    mode_for(held),
                    test["y"].to_numpy(),
                    res["p"],
                    res["tau"],
                    res["cost_tau"],
                )
            )
        rows.append(
            {
                "experiment": "E4",
                "model": name,
                "source": "macro",
                "mode": "macro",
                **_macro(rows[start:]),
            }
        )
    return {"note": NOTE, "rows": rows}


# --- E3/E4 mmBERT headline arms -------------------------------------------


def _fit_mmbert(
    train: pd.DataFrame,
    cal: pd.DataFrame,
    out_dir: str | Path,
    model_key: str,
    epochs: int = 3,
    max_len: int = 128,
    bs: int = 16,
) -> tuple[object, float, float, dict]:
    """Fine-tune one mmBERT checkpoint and freeze its threshold on ``cal_a``.

    Uses the same ``train_mmbert_pipeline`` as the final model, so temperature
    scaling and the FPR threshold follow the documented protocol. The extra
    cost threshold is recomputed on the same ``cal_a`` margins used for ``tau``.
    """
    from sklearn.model_selection import train_test_split

    from smishlens.calibrate import apply_temperature
    from smishlens.train import train_mmbert_pipeline
    from smishlens.train_mmbert import MMBertScorer, cal_margins

    if model_key not in MMBERT_MODELS:
        raise ValueError(f"unknown mmBERT model {model_key!r}")
    out = Path(out_dir)
    train_mmbert_pipeline(
        train,
        cal,
        out,
        max_fpr=MAX_FPR,
        seed=SEED,
        model_name=MMBERT_MODELS[model_key],
        model_label=f"mmbert_{model_key}_temp_scaled",
        epochs=epochs,
        max_len=max_len,
        bs=bs,
    )
    meta = json.loads((out / "metadata.json").read_text())
    scorer = MMBertScorer.load(out / "mmbert", T=meta["calibration"]["T"])
    cal_a, _ = train_test_split(cal, test_size=0.5, stratify=cal["y"], random_state=SEED)
    margins = cal_margins(out / "mmbert", cal_a["text"].tolist(), max_len=max_len)
    p_a = apply_temperature(margins, meta["calibration"]["T"])
    cost_tau = float(
        cost_operating_point(cal_a["y"].to_numpy(), p_a, fp_cost=10.0, fn_cost=1.0)["tau"]
    )
    return scorer, float(meta["threshold"]["value"]), cost_tau, meta


def run_e3_mmbert(
    frame: pd.DataFrame,
    model_key: str,
    out_dir: str | Path,
    epochs: int = 3,
    max_len: int = 128,
    bs: int = 16,
) -> dict:
    """mmBERT headline arm on the frozen pool test split.

    This is deliberately separate from ``run_e3``: a transformer has no
    train-OOF pass, so it cannot enter the classical SEL ranking. The model is
    fixed by the roadmap, trained on pool train, calibrated on cal, and the test
    split is scored once.
    """
    train = frame[frame["split"] == "train"]
    cal = frame[frame["split"] == "cal"]
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    scorer, tau, cost_tau, meta = _fit_mmbert(
        train, cal, out_dir, model_key, epochs=epochs, max_len=max_len, bs=bs
    )
    p = scorer.predict_proba(test["text"].astype(str).tolist())[:, 1]
    rows: list[dict] = []
    for src in sorted(test["source"].unique()):
        mask = (test["source"] == src).to_numpy()
        rows.append(
            _row(
                {"experiment": "E3", "model": f"mmbert_{model_key}", "source": src},
                mode_for(src),
                test.loc[mask, "y"].to_numpy(),
                p[mask],
                tau,
                cost_tau,
            )
        )
    rows.append(
        {
            "experiment": "E3",
            "model": f"mmbert_{model_key}",
            "source": "macro",
            "mode": "macro",
            **_macro(rows),
        }
    )
    return {
        "note": NOTE,
        "scope": "mmBERT headline arm; trained on pool train, calibrated on cal, scored once on pool test",
        "model": f"mmbert_{model_key}",
        "model_path": MMBERT_MODELS[model_key],
        "selection": "roadmap-fixed headline; no train-OOF pass exists for a transformer, so it is not part of the classical SEL",
        "threshold": {
            "tau": tau,
            "cost_tau": cost_tau,
            "policy": f"max recall s.t. FPR <= {MAX_FPR} on cal_a",
        },
        "calibration": meta.get("calibration"),
        "versions": {"torch": meta.get("torch"), "transformers": meta.get("transformers")},
        "n_train": int(len(train)),
        "n_cal": int(len(cal)),
        "n_test": int(len(test)),
        "rows": rows,
    }


def run_e4_mmbert(
    frame: pd.DataFrame,
    model_key: str,
    out_root: str | Path,
    epochs: int = 3,
    max_len: int = 128,
    bs: int = 16,
) -> dict:
    """Five-fold LOSO for one mmBERT checkpoint.

    Each fold trains on pool train rows excluding the held-out source and
    calibrates on cal rows excluding the same source, so the held-out source is
    never seen. The held-out source is scored at every frozen split label, as in
    the classical LOSO harness.
    """
    rows: list[dict] = []
    for held in HELD_OUT:
        train, test = _fold(frame, held)
        cal = frame[(frame["split"] == "cal") & (frame["source"] != held)]
        out_dir = Path(out_root) / f"heldout_{held}"
        scorer, tau, cost_tau, _ = _fit_mmbert(
            train, cal, out_dir, model_key, epochs=epochs, max_len=max_len, bs=bs
        )
        p = scorer.predict_proba(test["text"].astype(str).tolist())[:, 1]
        rows.append(
            _row(
                {
                    "experiment": "E4",
                    "model": f"mmbert_{model_key}",
                    "source": held,
                    "n_train": int(len(train)),
                    "n_cal": int(len(cal)),
                },
                mode_for(held),
                test["y"].to_numpy(),
                p,
                tau,
                cost_tau,
            )
        )
    rows.append(
        {
            "experiment": "E4",
            "model": f"mmbert_{model_key}",
            "source": "macro",
            "mode": "macro",
            **_macro(rows),
        }
    )
    return {
        "note": NOTE,
        "model": f"mmbert_{model_key}",
        "model_path": MMBERT_MODELS[model_key],
        "calibration": "cal rows excluding the held-out source, split into cal_a/cal_b exactly as train_mmbert_pipeline",
        "rows": rows,
    }


# --- E6 -------------------------------------------------------------------


def _e6_fold(models, held, frame, uci) -> list[dict]:
    """All models and both pool arms for one held-out source.

    Runs in its own process, so it rebuilds the two vectorisers it needs. The
    fold cache is built once and shared by every model and both arms, which is
    the whole point: the vectoriser costs 7.5 s and a fit costs 0.2-0.4 s.
    """
    pool_train, test = _fold(frame, held)
    uci_mats = build_fold_matrices(uci)
    mats = build_fold_matrices(pool_train, test)
    mode = mode_for(held)
    y = test["y"].to_numpy()
    weights = source_weights(pool_train)
    rows = []
    for name in models:
        oof = oof_cached(uci_mats, name)
        tau = float(threshold_for_fpr(uci_mats.y, oof, MAX_FPR)[0])
        cost_tau = float(cost_operating_point(uci_mats.y, oof, fp_cost=10.0, fn_cost=1.0)["tau"])
        est = _est(name).fit(uci_mats.X, uci_mats.y)
        p = _score_est(est, transform_with(uci_mats, test["text"]))
        rows.append(
            _row(
                {"experiment": "E6", "model": name, "arm": "uci_only",
                 "source": held, "n_train": len(uci)},
                mode, y, p, tau, cost_tau,
            )
        )
        for arm, w in (("pool_unweighted", None), ("pool_source_weighted", weights)):
            res = evaluate_cached(mats, name, w)
            rows.append(
                _row(
                    {"experiment": "E6", "model": name, "arm": arm,
                     "source": held, "n_train": len(pool_train)},
                    mode, y, res["p"], res["tau"], res["cost_tau"],
                )
            )
    return rows


def run_e6(
    frame: pd.DataFrame,
    uci: pd.DataFrame,
    models: list[str],
    headline: str,
    n_jobs: int = 6,
    with_gold: bool = True,
) -> dict:
    """UCI-only vs two pool arms on the same LOSO folds, then gold once.

    Parallelism is over folds, not over (model, fold). Two reasons, both
    measured: threads share liblinear's RNG state and shifted LinearSVC scores
    by ~1e-6 between identical runs, so the workers have to be processes; and a
    process per (model, fold) would refit the 7.5 s vectoriser five times for
    the same wall clock. One process per fold keeps the cache shared by all
    five models and both arms.
    """
    folds = [h for h in HELD_OUT]
    per_fold = Parallel(n_jobs=min(n_jobs, len(folds)), prefer="processes")(
        delayed(_e6_fold)(models, held, frame, uci) for held in folds
    )
    by_fold = dict(zip(folds, per_fold, strict=True))

    rows: list[dict] = []
    for name in models:
        block = [r for held in folds for r in by_fold[held] if r["model"] == name]
        rows.extend(block)
        for arm in ("uci_only", "pool_unweighted", "pool_source_weighted"):
            part = [r for r in block if r["arm"] == arm]
            rows.append(
                {"experiment": "E6", "model": name, "arm": arm,
                 "source": "macro", "mode": "macro", **_macro(part)}
            )

    paired = _paired(rows)
    gold = score_gold(frame, uci, headline) if with_gold else None
    return {"note": NOTE, "rows": rows, "paired": paired, "gold": gold}


def _paired(rows: list[dict]) -> list[dict]:
    out = []
    by_key = {(r["model"], r.get("arm"), r["source"]): r for r in rows if r["mode"] != "macro"}
    for model in sorted({r["model"] for r in rows}):
        for src in HELD_OUT:
            base = by_key.get((model, "uci_only", src))
            if not base:
                continue
            entry = {"model": model, "source": src, "mode": base["mode"]}
            for arm in ("pool_unweighted", "pool_source_weighted"):
                other = by_key.get((model, arm, src))
                if not other:
                    continue
                for metric in ("recall", "fpr", "pr_auc"):
                    if metric in base and metric in other:
                        entry[f"{arm}_{metric}"] = other[metric]
                        entry[f"delta_{arm}_{metric}"] = other[metric] - base[metric]
            out.append(entry)
    return out


# --- gold (once) ----------------------------------------------------------


def _group_rates(gold: pd.DataFrame, p: np.ndarray, tau: float, column: str) -> list[dict]:
    out = []
    for value, part in gold.groupby(column):
        idx = part.index.to_numpy()
        y = part["y"].to_numpy()
        pred = p[idx] >= tau
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        out.append(
            {
                column: value,
                "n": int(len(part)),
                "n_pos": pos,
                "n_neg": neg,
                "recall": float((pred & (y == 1)).sum() / pos) if pos else None,
                "fpr": float((pred & (y == 0)).sum() / neg) if neg else None,
            }
        )
    return out


def score_gold(
    frame: pd.DataFrame,
    uci: pd.DataFrame,
    models: str | list[str],
    uci_mats: FoldMatrices | None = None,
) -> dict:
    """One pass over the synthetic gold file with frozen train-only models.

    The headline and Dummy are scored together so the file is written once, and
    the gold labels are read only after the model and its thresholds were fixed
    from train-split evidence.
    """
    if isinstance(models, str):
        models = [models]
    gold = load_gold()
    pool_train = frame[frame["split"] == "train"]
    mats_by_arm = {
        "uci_only": uci_mats or build_fold_matrices(uci),
        "pool_unweighted": build_fold_matrices(pool_train),
    }
    y_gold = gold["y"].to_numpy()
    out = {}
    for model in models:
        arms = {}
        for arm, mats in mats_by_arm.items():
            oof = oof_cached(mats, model)
            tau = float(threshold_for_fpr(mats.y, oof, MAX_FPR)[0])
            est = _est(model).fit(mats.X, mats.y)
            p = _score_est(est, transform_with(mats, gold["text"]))
            m = metrics_at(y_gold, p, tau)
            arms[arm] = {
                "model": model,
                "tau": tau,
                "n_train": int(len(mats.y)),
                "n": int(len(gold)),
                "n_pos": m["n_pos"],
                "n_neg": m["n_neg"],
                "recall": m["recall"],
                "fpr": m["fpr"],
                "pr_auc": m["pr_auc"],
                "by_language": _group_rates(gold, p, tau, "language"),
                "by_scam_type": _group_rates(gold, p, tau, "scam_type"),
            }
        out[model] = {"arms": arms}
    return {
        "synthetic": True,
        "note": "fully synthetic gold file, written by Claude on the user's behalf; scored once, never used for tuning or selection",
        "scored_once": True,
        "models": out,
    }


# --- CLI ------------------------------------------------------------------


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="E5,E3,E4,E6")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--headline", default=None, help="model for the gold pass; defaults to the E3 selection")
    ap.add_argument("--n-jobs", type=int, default=6, help="threads for E6, parallel over (model, fold)")
    ap.add_argument(
        "--skip-gold",
        action="store_true",
        help="E6 without the gold pass; only for the n_jobs equivalence check, never for a result",
    )
    ap.add_argument("--mmbert-e3", choices=("base", "small"), default=None)
    ap.add_argument("--mmbert-e4", choices=("base", "small"), default=None)
    ap.add_argument("--mmbert-epochs", type=int, default=3)
    ap.add_argument("--mmbert-out", default=str(ROOT / "artifacts" / "mmbert"))
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)

    only = [x.strip().upper() for x in args.only.split(",") if x.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out = Path(args.out)
    frame = load_pool()

    if "E5" in only:
        e5 = run_e5(frame)
        _json(out / "e5_shortcut.json", e5)
        print(f"E5 pooled format AUC {e5['format_only_auc_pooled']:.3f} "
              f"length {e5['length_only_auc_pooled']:.3f} "
              f"imc25-vs-rest {e5['imc25_vs_rest_auc_from_format']:.3f}")

    headline = args.headline
    if "SEL" in only:
        sel = run_selection(frame, models)
        _json(out / "e3_selection.json", sel)
        print(f"selected {sel['selected_model']} by train-OOF macro PR-AUC")

    if "E3" in only:
        e3, headline = run_e3(frame, models)
        _json(out / "e3_pool.json", e3)
        print(f"E3 selected {headline} by train-OOF macro PR-AUC")
    if headline is None:
        # The headline is a written decision, not a fallback. e3_selection.json is
        # produced from train-split OOF only, before any gold row is read.
        for name in ("e3_selection.json", "e3_pool.json"):
            previous = out / name
            if previous.exists():
                headline = json.loads(previous.read_text())["selected_model"]
                print(f"E6 headline from {previous.name}: {headline}")
                break
        else:
            raise SystemExit(
                "no train-only selection found: run --only SEL first, "
                "gold must never pick the model"
            )

    if "E4" in only:
        _json(out / "e4_loso.json", run_e4(frame, models))

    if args.mmbert_e3:
        payload = run_e3_mmbert(
            frame,
            args.mmbert_e3,
            Path(args.mmbert_out) / f"e3_{args.mmbert_e3}",
            epochs=args.mmbert_epochs,
        )
        _json(out / f"e3_pool_mmbert_{args.mmbert_e3}.json", payload)
        macro = next(r for r in payload["rows"] if r["source"] == "macro")
        print(
            f"E3 mmBERT-{args.mmbert_e3} macro recall {macro.get('macro_recall'):.4f} "
            f"FPR {macro.get('macro_fpr'):.4f} PR-AUC {macro.get('macro_pr_auc'):.4f}"
        )

    if args.mmbert_e4:
        payload = run_e4_mmbert(
            frame,
            args.mmbert_e4,
            Path(args.mmbert_out) / f"e4_{args.mmbert_e4}",
            epochs=args.mmbert_epochs,
        )
        _json(out / f"e4_loso_mmbert_{args.mmbert_e4}.json", payload)
        macro = next(r for r in payload["rows"] if r["source"] == "macro")
        print(
            f"E4 mmBERT-{args.mmbert_e4} macro recall {macro.get('macro_recall'):.4f} "
            f"FPR {macro.get('macro_fpr'):.4f} PR-AUC {macro.get('macro_pr_auc'):.4f}"
        )

    if "GOLD" in only:
        # One pass, one write, immediately: nothing else runs before the file
        # exists. The headline was fixed from train-split evidence in
        # e3_selection.json before any gold row was read.
        gold = score_gold(frame, load_uci_train(), [headline, "dummy"])
        _json(out / "e6_gold.json", gold)
        print(f"gold scored once for {', '.join(gold['models'])} (synthetic)")
        return 0

    if "E6" in only:
        e6 = run_e6(
            frame, load_uci_train(), models, headline,
            n_jobs=args.n_jobs, with_gold=False,
        )
        _json(out / "e6_paired.json", e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
