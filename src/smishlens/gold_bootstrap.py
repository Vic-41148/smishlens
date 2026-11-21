"""Bootstrap 95% CIs for gold v1/v2 recall, FPR and PR-AUC.

Per-row gold scores were not saved (deliberately text-free), so the frozen
models are re-scored here with no refit. That is a deterministic repeat of the
gold pass, not a new look. Output is aggregates only: no text, ids or notes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from smishlens.evaluate import metrics_at, threshold_for_fpr
from smishlens.gold_v2 import GOLD_V1, GOLD_V2, MMBERT_E3, read_gold
from smishlens.pool_experiments import (
    MAX_FPR,
    RESULTS,
    _est,
    _score_est,
    build_fold_matrices,
    load_pool,
    oof_cached,
    transform_with,
)

N_RESAMPLES = 1000
SEED = 2026


def _scores(gold, mats) -> dict[str, tuple[np.ndarray, float]]:
    out = {}
    for name in ("svm", "dummy"):
        oof = oof_cached(mats, name)
        tau = float(threshold_for_fpr(mats.y, oof, MAX_FPR)[0])
        est = _est(name).fit(mats.X, mats.y)
        p = _score_est(est, transform_with(mats, gold["text"]))
        out[name] = (np.asarray(p, dtype=float), tau)
    return out


def _mmbert_scores(gold) -> tuple[np.ndarray, float]:
    from smishlens.train_mmbert import MMBertScorer

    meta = json.loads((MMBERT_E3 / "metadata.json").read_text())
    T = float(meta["calibration"]["T"])
    tau = float(meta["threshold"]["value"])
    scorer = MMBertScorer.load(MMBERT_E3 / "mmbert", T=T)
    return np.asarray(scorer.predict_proba(gold["text"].tolist())[:, 1]), tau


def _percentile_ci(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "n_valid": int(arr.size)}


def _bootstrap(y: np.ndarray, p: np.ndarray, tau: float, rng: np.random.Generator) -> dict:
    draws = {"recall": [], "fpr": [], "pr_auc": []}
    n = len(y)
    for _ in range(N_RESAMPLES):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        m = metrics_at(yy, pp, tau)
        for key in draws:
            value = m[key]
            if value == value:  # skip NaN on a degenerate resample
                draws[key].append(float(value))
    point = metrics_at(y, p, tau)
    out = {}
    for key, values in draws.items():
        out[key] = {"point": float(point[key]), **_percentile_ci(values)}
    return out


def run() -> dict:
    frame = load_pool()
    mats = build_fold_matrices(frame[frame["split"] == "train"])
    rng = np.random.default_rng(SEED)

    golds = {}
    for label, path in (("v1", GOLD_V1), ("v2", GOLD_V2)):
        gold = read_gold(path)
        y = gold["y"].to_numpy()
        models = _scores(gold, mats)
        p_mm, tau_mm = _mmbert_scores(gold)
        models["mmbert_base"] = (p_mm, tau_mm)
        golds[label] = {
            "n": int(len(gold)),
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
            "models": {
                name: _bootstrap(y, p, tau, rng) for name, (p, tau) in models.items()
            },
        }

    return {
        "method": "percentile bootstrap, 1000 resamples, fixed seed",
        "seed": SEED,
        "n_resamples": N_RESAMPLES,
        "note": (
            "frozen models re-scored with no refit because per-row gold scores "
            "were not saved; aggregates only, no text or ids"
        ),
        "gold": golds,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "e6_gold_v2_bootstrap.json"))
    args = ap.parse_args(argv)
    payload = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    for label, block in payload["gold"].items():
        for name, m in block["models"].items():
            print(
                f"gold {label} {name}: "
                f"recall {m['recall']['point']:.4f} [{m['recall']['lo']:.4f},{m['recall']['hi']:.4f}] "
                f"fpr {m['fpr']['point']:.4f} [{m['fpr']['lo']:.4f},{m['fpr']['hi']:.4f}] "
                f"pr_auc {m['pr_auc']['point']:.4f} [{m['pr_auc']['lo']:.4f},{m['pr_auc']['hi']:.4f}]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
