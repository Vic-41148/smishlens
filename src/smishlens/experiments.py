"""Tier-1 experiment runner (E1 preprocessing, E2 features, E3 models).

Everything here runs on the **train split only**, with 5-fold cross-validation.
The test split is never opened. Accuracy is never a scoring metric.

Usage
-----
    python -m smishlens.experiments                      # all three, defaults
    python -m smishlens.experiments --only E1 --out reports/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from smishlens.features import build_features

SEED = 42
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)
SCORING = {
    "ap": "average_precision",
    "f1": "f1",
    "mcc": make_scorer(matthews_corrcoef),
    "roc_auc": "roc_auc",
}
METRICS = ("ap", "f1", "mcc", "roc_auc")

# fixed, documented hyper-parameters: the ablation varies ONE thing at a time
LR = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED)


def _lr_pipe(feat_kw: dict) -> Pipeline:
    return Pipeline([("feats", build_features(**feat_kw)), ("clf", LR)])


# --------------------------------------------------------------------------- E1
E1 = [
    ("raw (no normaliser)", {"normalise_on": False}),
    ("masked (default)", {}),
    ("masked + stopwords", {"stopwords": True}),
    ("masked + stem", {"stem": True}),
    ("masked + deleet", {"deleet": True}),
    ("masked + min_df=1", {"min_df_word": 1, "min_df_char": 1}),
    ("masked + min_df=2", {"min_df_word": 2, "min_df_char": 2}),
    ("masked + min_df=3", {"min_df_word": 3, "min_df_char": 3}),
    ("masked, global confusables", {"confusables": "global"}),
]

# --------------------------------------------------------------------------- E2
E2 = [
    ("BoW (word 1-2)", {"blocks": ("bow",)}),
    ("word TF-IDF", {"blocks": ("word",)}),
    ("char TF-IDF", {"blocks": ("char",)}),
    ("word + char", {"blocks": ("word", "char")}),
    ("word + char + meta", {}),
]

# --------------------------------------------------------------------------- E3
E3 = [
    ("Dummy (prior)", DummyClassifier(strategy="prior")),
    ("MultinomialNB", MultinomialNB(alpha=0.1)),
    ("ComplementNB", ComplementNB(alpha=0.1)),
    ("LogisticRegression", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED)),
    ("LinearSVC", LinearSVC(C=0.5, class_weight="balanced", dual="auto", random_state=SEED)),
    (
        "RandomForest",
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=SEED,
        ),
    ),
]


def _evaluate(pipe: Pipeline, X, y) -> dict[str, float]:
    scores = cross_validate(pipe, X, y, cv=CV, scoring=SCORING, n_jobs=-1, error_score="raise")
    row: dict[str, float] = {}
    for metric in METRICS:
        values = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = float(np.mean(values))
        row[f"{metric}_std"] = float(np.std(values))
    return row


def _table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.sort_values("ap_mean", ascending=False).reset_index(drop=True)


def run_e1(X, y) -> pd.DataFrame:
    rows = []
    for name, feat_kw in E1:
        row = {"experiment": "E1", "variant": name, **_evaluate(_lr_pipe(feat_kw), X, y)}
        rows.append(row)
        print(f"  E1 {name:<28} AP={row['ap_mean']:.4f} ± {row['ap_std']:.4f}")
    return _table(rows)


def run_e2(X, y) -> pd.DataFrame:
    rows = []
    for name, feat_kw in E2:
        feats = build_features(**feat_kw).fit(X, y)
        n_feats = len(feats.get_feature_names_out())
        row = {
            "experiment": "E2",
            "variant": name,
            "n_features": n_feats,
            **_evaluate(_lr_pipe(feat_kw), X, y),
        }
        rows.append(row)
        print(f"  E2 {name:<28} AP={row['ap_mean']:.4f} ± {row['ap_std']:.4f}")
    return _table(rows)


def run_e3(X, y) -> pd.DataFrame:
    rows = []
    for name, clf in E3:
        pipe = Pipeline([("feats", build_features()), ("clf", clf)])
        row = {"experiment": "E3", "variant": name, **_evaluate(pipe, X, y)}
        rows.append(row)
        print(f"  E3 {name:<28} AP={row['ap_mean']:.4f} ± {row['ap_std']:.4f}")
    return _table(rows)


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, rule]
    for _, r in df.iterrows():
        cells = []
        for col in cols:
            value = r[col]
            if col.endswith("_mean"):
                metric = col[: -len("_mean")]
                cells.append(f"{value:.4f} ± {r[metric + '_std']:.4f}")
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_report(out: Path, tables: dict[str, pd.DataFrame], meta: dict) -> None:
    parts = [
        "# Tier-1 experiments — UCI anchor, train split only",
        "",
        f"- Train rows: **{meta['n_rows']:,}** "
        f"({meta['n_spam']:,} spam / {meta['n_rows'] - meta['n_spam']:,} ham, "
        f"prevalence {meta['prevalence']:.3f})",
        f"- Cross-validation: StratifiedKFold(5, shuffle=True, seed={SEED})",
        "- Metrics: PR-AUC (`ap`), F1, MCC, ROC-AUC. **Accuracy is not reported.**",
        "- The test split was not opened. All numbers are 5-fold CV means ± std.",
        "- E1 fixes LogisticRegression(C=1, class_weight=balanced); E2 fixes the same LR.",
        "- E3 uses the production feature set (word + char TF-IDF + meta).",
        "- Fold-to-fold std is shown next to every mean. Gaps smaller than roughly one "
        "std are not interpretable; UCI contains almost no obfuscation, so the E1 arms "
        "are expected to tie on this anchor.",
        "",
    ]
    labels = {
        "E1": "E1 — preprocessing ablation (same LR)",
        "E2": "E2 — feature ablation (same LR)",
        "E3": "E3 — classical model comparison",
    }
    cols = ["variant", "ap_mean", "f1_mean", "mcc_mean", "roc_auc_mean"]
    for key in ("E1", "E2", "E3"):
        if key not in tables:
            continue
        df = tables[key]
        parts += [f"## {labels[key]}", ""]
        if "n_features" in df.columns:
            parts.append(_md_table(df, cols + ["n_features"]))
        else:
            parts.append(_md_table(df, cols))
        best = df.iloc[0]
        parts += ["", f"**Best by PR-AUC:** {best['variant']} ({best['ap_mean']:.4f}).", ""]
    if "E3" in tables:
        champ = tables["E3"].iloc[0]["variant"]
        parts += [
            "## Champion",
            "",
            f"**{champ}** wins the train-split comparison on PR-AUC. "
            "It is promoted to the final retrain in C3, subject to the same "
            "threshold/conformal protocol.",
            "",
        ]
    (out / "tier1_report.md").write_text("\n".join(parts))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Tier-1 E1/E2/E3 on the UCI train split.")
    ap.add_argument("--train", default="data/processed/uci/train.csv")
    ap.add_argument("--out", default="reports/results")
    ap.add_argument("--only", choices=["E1", "E2", "E3"], action="append")
    args = ap.parse_args(argv)

    train_path = Path(args.train)
    if "test" in train_path.name.lower():
        raise SystemExit("refusing to run experiments on a test split")
    df = pd.read_csv(train_path, encoding_errors="replace")
    df = df.dropna(subset=["text", "y"])
    X = df["text"].astype(str).tolist()
    y = df["y"].astype(int).to_numpy()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "train_file": str(train_path),
        "n_rows": len(df),
        "n_spam": int(y.sum()),
        "prevalence": float(y.mean()),
        "seed": SEED,
        "cv_folds": CV.get_n_splits(),
    }
    print(f"{len(df)} rows, {int(y.sum())} spam, prevalence {y.mean():.3f}")

    wanted = args.only or ["E1", "E2", "E3"]
    tables: dict[str, pd.DataFrame] = {}
    for key, fn in (("E1", run_e1), ("E2", run_e2), ("E3", run_e3)):
        if key not in wanted:
            continue
        tables[key] = fn(X, y)
        tables[key].to_csv(out / f"tier1_{key.lower()}.csv", index=False)

    _write_report(out, tables, meta)
    summary = {
        "meta": meta,
        "best": {k: {"variant": v.iloc[0]["variant"], "ap": float(v.iloc[0]["ap_mean"])}
                 for k, v in tables.items()},
        "champion": tables["E3"].iloc[0]["variant"] if "E3" in tables else None,
    }
    (out / "tier1_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}/tier1_*.csv, tier1_report.md, tier1_summary.json")


if __name__ == "__main__":
    main()
