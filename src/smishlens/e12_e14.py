"""E12-E14: prevalence PPV, calibration, and conformal coverage under shift.

Every number here is derived from results already frozen in
``reports/results/`` or from the calibration split; no test split is scored
again for a decision, and gold is not touched.

- E12: arithmetic from frozen E3 (in-pool) and E4 (LOSO) recall/FPR.
- E13: ECE/Brier/reliability on ``cal_b`` only. The pool svm exposes a raw
  decision function, so Platt is fitted on ``cal_a`` and evaluated on
  ``cal_b``; the mmBERT temperature was already fitted on ``cal_a``.
- E14: conformal coverage/abstain in-pool (cal_a -> cal_b, plus 20 resplits on
  the calibration split) and under LOSO for moz/mishra. LOSO re-scores the
  frozen per-fold checkpoints with no refit, using each fold's frozen q_hat;
  that is a deterministic repeat, not a new look at a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from smishlens.calibrate import apply_platt, brier, ece, fit_platt, reliability
from smishlens.conformal import conformal_set, coverage, fit_mondrian, resplit_coverage
from smishlens.evaluate import ppv
from smishlens.pool_experiments import (
    RESULTS,
    ROOT,
    _est,
    _fold,
    build_fold_matrices,
    load_pool,
    transform_with,
)

FIGURES = ROOT / "reports" / "figures"
E3_CLASSICAL = RESULTS / "e3_pool.json"
E3_MMBERT = RESULTS / "e3_pool_mmbert_base.json"
E4_CLASSICAL = RESULTS / "e4_loso.json"
E4_MMBERT = RESULTS / "e4_loso_mmbert_small.json"
E3_MMBERT_DIR = ROOT / "artifacts" / "mmbert" / "e3_base"
E4_MMBERT_DIR = ROOT / "artifacts" / "mmbert" / "e4_small"
PREVALENCES = (0.01, 0.05, 0.10)
LOSO_SOURCES = ("mishra", "moz")


def _read_rows(path: Path, models: tuple[str, ...], sources: tuple[str, ...]) -> dict:
    data = json.loads(path.read_text())
    out = {}
    for row in data["rows"]:
        if row.get("model") in models and row.get("source") in sources:
            out[(row["model"], row["source"])] = row
    return out


# --- E12 ------------------------------------------------------------------


def _ppv_row(row: dict) -> dict:
    out = {"n_pos": row["n_pos"], "n_neg": row["n_neg"], "recall": row["recall"], "fpr": row["fpr"]}
    for prev in PREVALENCES:
        out[f"ppv_{int(prev * 100)}pct"] = ppv(row["recall"], row["fpr"], prev)
    return out


def run_e12() -> dict:
    e3 = _read_rows(E3_CLASSICAL, ("svm",), LOSO_SOURCES)
    e3.update(_read_rows(E3_MMBERT, ("mmbert_base",), LOSO_SOURCES))
    e4 = _read_rows(E4_CLASSICAL, ("svm",), LOSO_SOURCES)
    e4.update(_read_rows(E4_MMBERT, ("mmbert_small",), LOSO_SOURCES))

    def block(rows: dict, model: str) -> dict:
        per_source = {src: _ppv_row(rows[(model, src)]) for src in LOSO_SOURCES}
        macro = {
            key: float(np.mean([per_source[src][key] for src in LOSO_SOURCES]))
            for key in ("recall", "fpr", *(f"ppv_{int(p * 100)}pct" for p in PREVALENCES))
        }
        return {"per_source": per_source, "macro": macro}

    return {
        "prevalences": list(PREVALENCES),
        "in_pool_e3": {
            "note": "E3 pool test split, frozen FPR<=1% thresholds; mmBERT is base",
            "svm": block(e3, "svm"),
            "mmbert_base": block(e3, "mmbert_base"),
        },
        "loso_e4": {
            "note": (
                "E4 five-fold LOSO. svm is classical; mmbert_small is the small "
                "checkpoint. mmBERT-base was NOT LOSO-tested; do not read this "
                "as a base-model result."
            ),
            "svm": block(e4, "svm"),
            "mmbert_small": block(e4, "mmbert_small"),
        },
    }


# --- E13 ------------------------------------------------------------------


def _cal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cal = frame[frame["split"] == "cal"]
    return train_test_split(cal, test_size=0.5, stratify=cal["y"], random_state=42)


def _calibration_block(p: np.ndarray, y: np.ndarray, method: str) -> dict:
    return {
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "method": method,
        "ece_10bin": ece(p, y, bins=10),
        "brier": brier(p, y),
        "mean_confidence": float(np.mean(p)),
        "base_rate": float(np.mean(y)),
        "reliability": reliability(p, y, bins=10),
    }


def run_e13(frame: pd.DataFrame) -> tuple[dict, dict]:
    cal_a, cal_b = _cal_split(frame)
    pool_train = frame[frame["split"] == "train"]

    mats = build_fold_matrices(pool_train)
    est = _est("svm").fit(mats.X, mats.y)
    s_a = est.decision_function(transform_with(mats, cal_a["text"]))
    s_b = est.decision_function(transform_with(mats, cal_b["text"]))
    a, b = fit_platt(s_a, cal_a["y"].to_numpy())
    p_svm = apply_platt(s_b, a, b)

    meta = json.loads((E3_MMBERT_DIR / "metadata.json").read_text())
    T = float(meta["calibration"]["T"])
    from smishlens.train_mmbert import MMBertScorer

    scorer = MMBertScorer.load(E3_MMBERT_DIR / "mmbert", T=T)
    p_mmbert = scorer.predict_proba(cal_b["text"].tolist())[:, 1]

    y_b = cal_b["y"].to_numpy()
    payload = {
        "split": "cal_b only; cal_a fits the map, cal_b is evaluated",
        "note": (
            "cal_b is also the half the E3 conformal q_hat was fitted on; it is "
            "held out for temperature/Platt, not unused"
        ),
        "models": {
            "svm_pool": _calibration_block(
                p_svm, y_b,
                f"Platt sigmoid(a*score+b) fitted on cal_a decision_function (a={a:.6f}, b={b:.6f})",
            ),
            "mmbert_base": _calibration_block(
                p_mmbert, y_b, f"temperature scaling, T={T:.6f}, fitted on cal_a",
            ),
        },
    }
    return payload


def plot_reliability(blocks: dict, path: Path) -> None:
    """Equal-width reliability bins, with sparse bins shown but not connected.

    Almost all calibration mass sits in the two extreme bins, so a line through
    one-row mid bins would zigzag and imply structure that is not there.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color="0.55", lw=1, ls="--", label="perfect")
    for name, block in blocks.items():
        bins = block["reliability"]
        xs = [row["confidence"] for row in bins]
        ys = [row["frequency"] for row in bins]
        ns = [row["n"] for row in bins]
        sizes = [8 + 90 * (n / max(ns)) ** 0.5 for n in ns]
        ax.scatter(xs, ys, s=sizes, alpha=0.55, label=name, zorder=3)
        dense_x = [x for x, n in zip(xs, ns, strict=True) if n >= 20]
        dense_y = [y for y, n in zip(ys, ns, strict=True) if n >= 20]
        ax.scatter(
            dense_x, dense_y, s=70, facecolors="none", edgecolors="black",
            linewidths=0.8, zorder=4,
        )
    ax.set_xlabel("mean predicted P(spam)")
    ax.set_ylabel("observed spam rate")
    ax.set_title("E13 reliability on cal_b (marker size = bin count)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- E14 ------------------------------------------------------------------


def _coverage_block(p, y, qs: dict[int, float], label: str) -> dict:
    cov, abstain = coverage(p, y, qs)
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    pred_sets = [conformal_set(float(x), qs) for x in p]
    overall = float(np.mean([t in s for s, t in zip(pred_sets, y, strict=True)]))
    return {
        "label": label,
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "coverage_ham": cov.get(0),
        "coverage_spam": cov.get(1),
        "coverage_overall": overall,
        "abstain_rate": abstain,
        "q_hat": {str(k): float(v) for k, v in qs.items()},
    }


def _resplit_summary(rows: list[dict]) -> dict:
    keys = ("cov_ham", "cov_spam", "abstain")
    return {
        "n_splits": len(rows),
        "mean": {k: float(np.nanmean([r[k] for r in rows])) for k in keys},
        "min": {k: float(np.nanmin([r[k] for r in rows])) for k in keys},
        "max": {k: float(np.nanmax([r[k] for r in rows])) for k in keys},
    }


def run_e14(frame: pd.DataFrame) -> dict:
    cal_a, cal_b = _cal_split(frame)
    pool_train = frame[frame["split"] == "train"]
    cal = frame[frame["split"] == "cal"]

    mats = build_fold_matrices(pool_train)
    est = _est("svm").fit(mats.X, mats.y)
    s_cal = est.decision_function(transform_with(mats, cal["text"]))
    s_a = est.decision_function(transform_with(mats, cal_a["text"]))
    a, b = fit_platt(s_a, cal_a["y"].to_numpy())
    p_cal_svm = apply_platt(s_cal, a, b)
    p_a_svm = apply_platt(s_a, a, b)
    p_b_svm = apply_platt(
        est.decision_function(transform_with(mats, cal_b["text"])), a, b
    )

    meta = json.loads((E3_MMBERT_DIR / "metadata.json").read_text())
    T = float(meta["calibration"]["T"])
    from smishlens.train_mmbert import MMBertScorer

    scorer = MMBertScorer.load(E3_MMBERT_DIR / "mmbert", T=T)
    p_cal_mmbert = scorer.predict_proba(cal["text"].tolist())[:, 1]
    p_a_mmbert = scorer.predict_proba(cal_a["text"].tolist())[:, 1]
    p_b_mmbert = scorer.predict_proba(cal_b["text"].tolist())[:, 1]

    y_cal = cal["y"].to_numpy()
    y_a = cal_a["y"].to_numpy()
    y_b = cal_b["y"].to_numpy()

    in_pool = {}
    for name, p_all, p_a, p_b in (
        ("svm_pool", p_cal_svm, p_a_svm, p_b_svm),
        ("mmbert_base", p_cal_mmbert, p_a_mmbert, p_b_mmbert),
    ):
        qs = fit_mondrian(p_a, y_a, alpha=0.05)
        in_pool[name] = {
            "point_estimate": _coverage_block(p_b, y_b, qs, "cal_a -> cal_b"),
            "resplits": _resplit_summary(resplit_coverage(p_all, y_cal, n_splits=20, seed=0)),
        }

    loso = {}
    for held in LOSO_SOURCES:
        fold_meta = json.loads((E4_MMBERT_DIR / f"heldout_{held}" / "metadata.json").read_text())
        T_fold = float(fold_meta["calibration"]["T"])
        qs = {int(k): float(v) for k, v in fold_meta["conformal"]["q_hat"].items()}
        scorer_fold = MMBertScorer.load(E4_MMBERT_DIR / f"heldout_{held}" / "mmbert", T=T_fold)
        _, test = _fold(frame, held)
        p = scorer_fold.predict_proba(test["text"].astype(str).tolist())[:, 1]
        loso[held] = _coverage_block(p, test["y"].to_numpy(), qs, f"LOSO {held}, frozen q_hat")

    return {
        "nominal_coverage": 0.95,
        "alpha": 0.05,
        "in_pool": {
            "note": (
                "cal_a fits q_hat, cal_b is evaluated; resplits draw 20 random "
                "50/50 splits of the calibration split"
            ),
            **in_pool,
        },
        "loso": {
            "note": (
                "mmBERT-small frozen fold checkpoints, no refit; q_hat read from "
                "each fold metadata.json. mmBERT-base was not LOSO-tested. "
                "Resplits do not apply: q_hat is frozen per fold."
            ),
            **loso,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="E12,E13,E14")
    args = ap.parse_args(argv)
    only = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    frame = load_pool() if ({"E13", "E14"} & only) else None

    if "E12" in only:
        payload = run_e12()
        out = RESULTS / "e12_ppv.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}")
    if "E13" in only:
        payload = run_e13(frame)
        out = RESULTS / "e13_calibration.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}")
        fig = FIGURES / "e13_reliability.png"
        plot_reliability(payload["models"], fig)
        print(f"wrote {fig}")
        for name, block in payload["models"].items():
            print(f"E13 {name} ECE={block['ece_10bin']:.4f} Brier={block['brier']:.4f}")
    if "E14" in only:
        payload = run_e14(frame)
        out = RESULTS / "e14_conformal.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}")
        for name, block in payload["in_pool"].items():
            if not isinstance(block, dict) or "point_estimate" not in block:
                continue
            pe = block["point_estimate"]
            print(
                f"E14 in-pool {name} cov_ham={pe['coverage_ham']:.3f} "
                f"cov_spam={pe['coverage_spam']:.3f} abstain={pe['abstain_rate']:.3f}"
            )
        for held, block in payload["loso"].items():
            if not isinstance(block, dict) or "coverage_ham" not in block:
                continue
            print(
                f"E14 LOSO {held} cov_ham={block['coverage_ham']:.3f} "
                f"cov_spam={block['coverage_spam']:.3f} abstain={block['abstain_rate']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
