"""Can formatting alone separate the classes, or identify the source? (RQ2, E5)

The risk this probe measures: a model that scores 99% by recognising *which
source* a message came from (era, OCR artefacts, quote style, casing) rather
than the scam. Formatting-only and length-only AUCs are the evidence. This is
never pass/fail; report them next to the content model's numbers.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

CURLY = "\u2018\u2019\u201c\u201d"
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]")


def format_features(texts) -> pd.DataFrame:
    """Twelve shape features that carry no lexical content."""
    rows = []
    for item in texts:
        s = str(item)
        n = max(len(s), 1)
        words = s.split()
        rows.append(
            {
                "n_chars": len(s),
                "n_words": len(words),
                "upper_ratio": sum(c.isupper() for c in s) / n,
                "punct_ratio": sum((not c.isalnum()) and (not c.isspace()) for c in s) / n,
                "digit_ratio": sum(c.isdigit() for c in s) / n,
                "curly_quotes": sum(c in CURLY for c in s),
                "double_space": int("  " in s),
                "nonascii_ratio": sum(not c.isascii() for c in s) / n,
                "ends_punct": int(s.rstrip()[-1:] in ".!?") if s.strip() else 0,
                "starts_upper": int(s[:1].isupper()),
                "has_newline": int("\n" in s),
                "trailing_space": int(s != s.rstrip()),
                "n_emoji": len(EMOJI.findall(s)),
            }
        )
    return pd.DataFrame(rows)


def _cv_auc(X, y, seed: int = 0) -> float:
    """Cross-validated AUC of a formatting model. NaN when it cannot be fitted."""
    y = np.asarray(y)
    n_classes = len(np.unique(y))
    smallest = int(np.bincount(y.astype(int)).min()) if n_classes > 1 else 0
    if n_classes < 2 or smallest < 5:
        return float("nan")
    proba = cross_val_predict(
        HistGradientBoostingClassifier(max_iter=80, random_state=seed),
        X,
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        method="predict_proba",
    )
    if proba.shape[1] == 2:
        return float(roc_auc_score(y, proba[:, 1]))
    return float(roc_auc_score(y, proba, multi_class="ovr"))


def shortcut_report(
    df: pd.DataFrame, min_per_class: int = 20, fpr_only: tuple[str, ...] = ()
) -> dict:
    """Pooled, per-source and source-identifiability formatting probes.

    ``fpr_only`` names ham-only sources. They have no positives, so a
    class-separation AUC cannot be estimated for them and they are kept out of
    the pooled AUC; mixing them in would turn "does this look like spam?" into
    "which source is this?" without saying so. They get a separate descriptive
    block instead. Source identifiability still uses every source, because the
    question there is exactly whether shape reveals the origin.
    """
    df = df.reset_index(drop=True)
    feats = format_features(df["text"])
    y = df["y"].values
    fpr_only = tuple(fpr_only)
    class_rows = ~df["source"].isin(fpr_only).to_numpy()
    class_feats = feats.loc[class_rows].reset_index(drop=True)
    class_y = y[class_rows]

    out: dict = {
        "format_only_auc": _cv_auc(class_feats, class_y),
        "length_only_auc": _cv_auc(class_feats[["n_chars"]], class_y),
        "class_auc_scope": "rows from sources with both classes present; fpr_only sources excluded",
        "by_source": {},
        "fpr_only": {},
    }
    for src, group in df.groupby("source"):
        if str(src) in fpr_only:
            continue
        if (group.y == 1).sum() >= min_per_class and (group.y == 0).sum() >= min_per_class:
            sub = feats.loc[group.index].reset_index(drop=True)
            out["by_source"][str(src)] = {
                "format_only_auc": _cv_auc(sub, group.y.values),
                "length_only_auc": _cv_auc(sub[["n_chars"]], group.y.values),
            }

    for src in fpr_only:
        group = df[df["source"] == src]
        if group.empty:
            continue
        sub = feats.loc[group.index]
        out["fpr_only"][src] = {
            "n_rows": int(len(group)),
            "n_pos": int((group["y"] == 1).sum()),
            "n_neg": int((group["y"] == 0).sum()),
            "class_auc": None,
            "why": "ham only; no class-separation AUC, FPR is reported by the LOSO runners",
            "mean_n_chars": float(sub["n_chars"].mean()),
            "mean_digit_ratio": float(sub["digit_ratio"].mean()),
            "mean_upper_ratio": float(sub["upper_ratio"].mean()),
            "share_with_double_space": float(sub["double_space"].mean()),
        }

    # Source identifiability: can formatting tell the sources apart?
    codes = df["source"].astype("category").cat.codes.values
    n_sources = int(codes.max()) + 1
    smallest = int(np.bincount(codes).min()) if n_sources else 0
    if n_sources > 1 and smallest >= 5:
        pred = cross_val_predict(
            HistGradientBoostingClassifier(max_iter=80, random_state=0),
            feats,
            codes,
            cv=StratifiedKFold(5, shuffle=True, random_state=0),
            method="predict",
        )
        out["source_id_acc_from_format"] = float((pred == codes).mean())
        out["source_id_chance"] = float(np.bincount(codes).max() / len(codes))
    else:
        out["source_id_acc_from_format"] = float("nan")
        out["source_id_chance"] = float("nan")
    out["source_id_sources"] = sorted(df["source"].unique().tolist())
    return out
