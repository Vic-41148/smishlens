"""Tier-2 experiment runner on the frozen modern pool: E3, E4, E5, E6.

Rules baked in (roadmap §8, and the 86% base rate):

- Per-source results plus a **macro average across sources**. Never a pooled
  micro-average, never accuracy, never precision/AP read off the pooled test.
- Both-class folds: **recall at FPR <= 1%**, with the threshold frozen on the
  training out-of-fold scores, plus **PR-AUC**. Recall-only folds: recall only.
- PPV is reported at 5 / 10 / 20% prevalence from that fold's recall and FPR.
- E6 compares three training arms on the same folds: UCI-only, modern pool
  unweighted, modern pool with source-balanced sample weights. Weighting only:
  the frozen split is untouched.

Usage
-----
    python -m smishlens.pool_experiments --only E5 --dry-run 500
    python -m smishlens.pool_experiments --only E3,E4,E6 --out reports/results

``--dry-run N`` samples at most N rows and writes no files. It exists to check
the code paths, not to produce numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from smishlens.evaluate import metrics_at, ppv, threshold_for_fpr
from smishlens.features import build_features
from smishlens.pool import BOTH_CLASSES  # single source of truth for fold kinds
from smishlens.probes import shortcut_report

ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = ROOT / "data" / "processed" / "modern"
UCI_DIR = ROOT / "data" / "processed" / "uci"
SEED = 42
MAX_FPR = 0.01
PREVALENCES = (0.05, 0.10, 0.20)
RECALL_ONLY = ("imc25", "smishtank")


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
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        )
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
    return pd.read_csv(UCI_DIR / "train.csv")


def source_weights(train: pd.DataFrame) -> np.ndarray:
    """1 / (rows of that source and class), rescaled to mean 1.

    Each (source, label) cell contributes the same total weight, so a 24k-row
    positives-only source cannot drown the ham sources. No rows are dropped.
    """
    cells = train.groupby(["source", "y"])["text"].transform("size")
    w = 1.0 / cells.to_numpy(dtype=float)
    return w / w.mean()


def _scores(y, p, tau) -> dict[str, float]:
    m = metrics_at(y, p, tau)
    out = {"n_pos": m["n_pos"], "n_neg": m["n_neg"], "recall": m["recall"]}
    if m["n_neg"] > 0:
        out["fpr"] = m["fpr"]
        out["pr_auc"] = m["pr_auc"]
        for prev in PREVALENCES:
            out[f"ppv_{int(prev * 100)}pct"] = ppv(m["recall"], m["fpr"], prev)
    return out


def _method(pipe: Pipeline) -> str:
    return "predict_proba" if hasattr(pipe, "predict_proba") else "decision_function"


def _decision(pipe: Pipeline, texts) -> np.ndarray:
    """Probability, or the decision value when the estimator has no proba."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(texts)[:, 1]
    return pipe.decision_function(texts)


def fit_threshold(train: pd.DataFrame, name: str, weights: np.ndarray | None = None):
    """Threshold from out-of-fold scores on the training rows only."""
    pipe = pipe_for(name)
    params = {} if weights is None else {"clf__sample_weight": weights}
    oof = cross_val_predict(
        pipe,
        train["text"],
        train["y"],
        method=_method(pipe),
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        params=params,
        n_jobs=1,
    )
    if oof.ndim == 2:
        oof = oof[:, 1]
    return float(threshold_for_fpr(train["y"].values, oof, MAX_FPR)[0])


def score_source(
    train: pd.DataFrame,
    test: pd.DataFrame,
    name: str,
    weights: np.ndarray | None = None,
) -> tuple[dict[str, float], float]:
    """Fit on ``train``, freeze tau on its OOF, score ``test`` once."""
    tau = fit_threshold(train, name, weights)
    pipe = pipe_for(name)
    params = {} if weights is None else {"clf__sample_weight": weights}
    pipe.fit(train["text"], train["y"], **params)
    p = _decision(pipe, test["text"])
    return _scores(test["y"].values, p, tau), tau


def _macro(rows: pd.DataFrame) -> dict[str, float]:
    both = rows[rows["mode"] == "both"]
    out = {"n_folds": float(len(both))}
    for key in ("recall", "fpr", "pr_auc", "ppv_5pct", "ppv_10pct", "ppv_20pct"):
        if key in both and both[key].notna().any():
            out[f"macro_{key}"] = float(both[key].mean())
    return out


def run_e5(frame: pd.DataFrame, min_per_class: int = 20) -> pd.DataFrame:
    report = shortcut_report(frame, min_per_class=min_per_class)
    rows = [{"source": "pooled_for_reference", "format_only_auc": report["format_only_auc"],
             "length_only_auc": report["length_only_auc"]}]
    for src, vals in report["by_source"].items():
        rows.append({"source": src, **vals})
    rows.append(
        {
            "source": "macro_over_sources",
            "format_only_auc": float(np.mean([v["format_only_auc"] for v in report["by_source"].values()])),
            "length_only_auc": float(np.mean([v["length_only_auc"] for v in report["by_source"].values()])),
        }
    )
    rows.append(
        {
            "source": "source_identifiability",
            "format_only_auc": report["source_id_acc_from_format"],
            "length_only_auc": report["source_id_chance"],
        }
    )
    return pd.DataFrame(rows)


def run_e3(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    rows = []
    for name in models:
        tau = fit_threshold(train, name)
        pipe = pipe_for(name).fit(train["text"], train["y"])
        p = _decision(pipe, test["text"])
        for src in sorted(test["source"].unique()):
            mask = (test["source"] == src).to_numpy()
            scores = _scores(test.loc[mask, "y"].values, p[mask], tau)
            mode = "both" if scores["n_neg"] > 0 else "recall_only"
            rows.append({"experiment": "E3", "model": name, "source": src, "mode": mode,
                         "tau": tau, **scores})
        block = pd.DataFrame(rows)
        block = block[(block["experiment"] == "E3") & (block["model"] == name)]
        rows.append({"experiment": "E3", "model": name, "source": "macro",
                     "mode": "macro", **_macro(block)})
    return pd.DataFrame(rows)


def _fold_frame(frame: pd.DataFrame, held: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[(frame["split"] == "train") & (frame["source"] != held)]
    test = frame[frame["source"] == held]
    return train, test


def score_with(pipe: Pipeline, tau: float, test: pd.DataFrame) -> dict[str, float]:
    """Score a pre-fitted model at a frozen threshold."""
    p = _decision(pipe, test["text"])
    return _scores(test["y"].values, p, tau)


def run_e4(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for name in models:
        for held in (*BOTH_CLASSES, *RECALL_ONLY):
            train, test = _fold_frame(frame, held)
            scores, tau = score_source(train, test, name)
            mode = "both" if scores["n_neg"] > 0 else "recall_only"
            rows.append({"experiment": "E4", "model": name, "source": held, "mode": mode,
                         "n_train": len(train), "tau": tau, **scores})
        block = pd.DataFrame(rows)
        block = block[(block["experiment"] == "E4") & (block["model"] == name)]
        rows.append({"experiment": "E4", "model": name, "source": "macro",
                     "mode": "macro", **_macro(block)})
    return pd.DataFrame(rows)


def run_e6(frame: pd.DataFrame, uci: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for name in models:
        # The UCI-only arm does not depend on the fold: fit and freeze it once.
        tau_uci = fit_threshold(uci, name)
        pipe_uci = pipe_for(name).fit(uci["text"], uci["y"])
        for held in (*BOTH_CLASSES, *RECALL_ONLY):
            pool_train, test = _fold_frame(frame, held)
            scores = score_with(pipe_uci, tau_uci, test)
            mode = "both" if scores["n_neg"] > 0 else "recall_only"
            rows.append({"experiment": "E6", "model": name, "arm": "uci_only",
                         "source": held, "mode": mode, "n_train": len(uci),
                         "tau": tau_uci, **scores})
            for arm, weights in (
                ("pool_unweighted", None),
                ("pool_source_weighted", source_weights(pool_train)),
            ):
                scores, tau = score_source(pool_train, test, name, weights)
                mode = "both" if scores["n_neg"] > 0 else "recall_only"
                rows.append({"experiment": "E6", "model": name, "arm": arm,
                             "source": held, "mode": mode, "n_train": len(pool_train),
                             "tau": tau, **scores})
        block = pd.DataFrame(rows)
        for arm in ("uci_only", "pool_unweighted", "pool_source_weighted"):
            part = block[(block["experiment"] == "E6") & (block["model"] == name)
                         & (block["arm"] == arm) & (block["source"] != "macro")]
            rows.append({"experiment": "E6", "model": name, "arm": arm,
                         "source": "macro", "mode": "macro", **_macro(part)})
    return pd.DataFrame(rows)


def e6_delta(table: pd.DataFrame) -> pd.DataFrame:
    """Paired gap per fold: modern pool minus UCI-only (recall and PR-AUC)."""
    keys = ["model", "source"]
    wide = table[table["mode"] != "macro"].pivot_table(
        index=keys, columns="arm", values=["recall", "pr_auc"], aggfunc="first"
    )
    rows = []
    for (model, source), row in wide.iterrows():
        out = {"model": model, "source": source}
        for metric in ("recall", "pr_auc"):
            uci = row.get((metric, "uci_only"), float("nan"))
            for arm in ("pool_unweighted", "pool_source_weighted"):
                out[f"{arm}_{metric}"] = row.get((metric, arm), float("nan"))
                out[f"delta_{arm}_{metric}"] = row.get((metric, arm), float("nan")) - uci
        rows.append(out)
    return pd.DataFrame(rows)


def dry_run_sample(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """At most ``n`` rows, spread across sources; keeps the frozen split labels."""
    per = max(n // frame["source"].nunique(), 1)
    parts = []
    for _, group in frame.groupby("source"):
        parts.append(group.sample(min(per, len(group)), random_state=SEED))
    return pd.concat(parts, ignore_index=True).head(n)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="E5,E3,E4,E6")
    ap.add_argument("--models", default="svm", help="E4/E6 models (E3 uses the full list)")
    ap.add_argument("--out", default=str(ROOT / "reports" / "results"))
    ap.add_argument("--dry-run", type=int, default=0, metavar="N")
    args = ap.parse_args(argv)

    only = [x.strip().upper() for x in args.only.split(",") if x.strip()]
    frame = load_pool()
    if args.dry_run:
        frame = dry_run_sample(frame, args.dry_run)
        print(f"DRY RUN on {len(frame)} rows — no result files will be written")

    out = Path(args.out)
    tables: dict[str, pd.DataFrame] = {}

    if "E5" in only:
        tables["E5"] = run_e5(frame, min_per_class=5 if args.dry_run else 20)
        print(tables["E5"].to_string(index=False))
    if "E3" in only:
        models = ["dummy", "mnb", "cnb", "lr", "svm", "rf"]
        tables["E3"] = run_e3(frame, models)
        print(tables["E3"].to_string(index=False))
    if "E4" in only:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        tables["E4"] = run_e4(frame, models)
        print(tables["E4"].to_string(index=False))
    if "E6" in only:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        tables["E6"] = run_e6(frame, load_uci_train(), models)
        tables["E6_delta"] = e6_delta(tables["E6"])
        print(tables["E6"].to_string(index=False))
        print(tables["E6_delta"].to_string(index=False))

    if args.dry_run:
        print("dry run complete — nothing written")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = out / f"tier2_{name.lower()}.csv"
        table.to_csv(path, index=False)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
