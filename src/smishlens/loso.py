"""Leave-one-source-out harness (RQ1, E4, E6).

``fit_predict(train_df, test_df) -> (p_spam_test, tau)`` is the only contract.
The threshold is frozen on the training sources, so the FPR seen on the held-out
source is the real cost of drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from smishlens.evaluate import metrics_at, threshold_for_fpr
from smishlens.features import build_final


def loso(
    df: pd.DataFrame,
    fit_predict,
    train_sources: list[str] | None = None,
    min_pos: int = 30,
    min_neg: int = 30,
    exclude: tuple[str, ...] = ("synthetic",),
) -> pd.DataFrame:
    """Hold out one source at a time.

    Modes: ``both`` (recall, FPR, AUCs), ``recall_only`` (a positives-only
    source such as Smishtank), ``fpr_only`` (a ham-only source such as ExAIS:
    it trains, but only its false-positive rate is meaningful), ``too_small``
    (reported, not scored). ``train_sources`` restricts the training pool, e.g.
    ``["uci"]`` for E6.
    """
    rows: list[dict] = []
    for src, test in df.groupby("source"):
        if src in exclude:
            continue
        train = df[df["source"] != src]
        if train_sources is not None:
            train = train[train["source"].isin(train_sources)]
        if train["y"].nunique() < 2:
            continue
        pos = int((test["y"] == 1).sum())
        neg = int((test["y"] == 0).sum())
        if pos >= min_pos and neg >= min_neg:
            mode = "both"
        elif neg == 0 and pos >= min_pos:
            mode = "recall_only"
        elif pos == 0 and neg >= min_neg:
            mode = "fpr_only"
        else:
            rows.append({"source": src, "mode": "too_small", "n_pos": pos, "n_neg": neg})
            continue

        p, tau = fit_predict(train, test)
        metrics = metrics_at(test["y"].values, p, tau)
        if mode == "recall_only":
            metrics = {k: metrics[k] for k in ("n_pos", "n_neg", "recall")}
        elif mode == "fpr_only":
            metrics = {k: metrics[k] for k in ("n_pos", "n_neg", "fpr")}
        rows.append({"source": src, "mode": mode, "tau": tau, "n_train": len(train), **metrics})
    return pd.DataFrame(rows)


def macro(res: pd.DataFrame) -> dict[str, float]:
    """Unweighted mean over ``both`` folds. Always print n beside it.

    ``fpr_only`` (ham-only) and ``recall_only`` folds are counted but never
    averaged: they cannot estimate the missing class, so admitting them would
    silently turn a missing metric into a zero.
    """
    both = res[res["mode"] == "both"]
    out: dict[str, float] = {
        "folds": float(len(both)),
        "n_fpr_only": int((res["mode"] == "fpr_only").sum()),
        "n_recall_only": int((res["mode"] == "recall_only").sum()),
    }
    for key in ("recall", "fpr", "pr_auc", "roc_auc"):
        if key in both and both[key].notna().any():
            value = float(both[key].mean())
            if np.isfinite(value):
                out[key] = value
    return out


RESULTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "results"


def _clean(value):
    """JSON-safe scalar: NaN/inf become null, numpy scalars become Python ones."""
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_results(
    res: pd.DataFrame,
    name: str = "loso",
    out_dir: Path | str = RESULTS_DIR,
    meta: dict | None = None,
) -> Path:
    """Write per-source rows plus the macro average to ``out_dir/<name>.json``.

    ``macro`` averages ``both`` folds only, so a positives-only or ham-only
    source cannot drag it. A metric that does not exist for a fold (``fpr`` on
    a recall-only source, ``recall`` on an ``fpr_only`` source) is left out of
    that row rather than written as null, so the file is valid JSON and each
    row reports only what its mode supports. ``by_mode`` lists which sources
    were treated as one-class, so a reader can see the separation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta or {},
        "by_mode": {
            mode: sorted(res.loc[res["mode"] == mode, "source"].astype(str).tolist())
            for mode in ("both", "recall_only", "fpr_only", "too_small")
        },
        "per_source": [
            {k: _clean(v) for k, v in row.items() if _clean(v) is not None}
            for row in res.to_dict(orient="records")
        ],
        "macro": {k: _clean(v) for k, v in macro(res).items()},
    }
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return path


def fit_predict_svm(train: pd.DataFrame, test: pd.DataFrame, C: float = 0.5):
    """Classical LOSO step: fit, threshold on out-of-fold train, score test once."""
    model = build_final(C).fit(train["text"], train["y"])
    oof = cross_val_predict(
        build_final(C),
        train["text"],
        train["y"],
        method="predict_proba",
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
    )[:, 1]
    tau = threshold_for_fpr(train["y"].values, oof, 0.01)[0]
    return model.predict_proba(test["text"])[:, 1], tau
