"""Second gold pass: one write, ``e6_gold_v2.json``, never the first file.

Both gold files are scored in separate blocks. Gold v1 was already scored
before the ExAIS ham-only rebuild, so this pass is a repeat for v1, not a
pristine first look. Gold v2 was frozen before any model saw it and is scored
exactly once here. Provenance is recorded as the SHA-256 of the file, not a
commit id: the commit that froze it was dropped from the public history on
2026-11-22 when the text-bearing splits were purged.

No row text, id or notes leaves this module: every block is an aggregate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from smishlens.evaluate import metrics_at, threshold_for_fpr
from smishlens.local_data import sha256
from smishlens.pool_experiments import (
    MAX_FPR,
    RESULTS,
    ROOT,
    _est,
    _group_rates,
    _score_est,
    build_fold_matrices,
    load_pool,
    oof_cached,
    transform_with,
)

GOLD_V1 = ROOT / "data" / "gold_holdout.csv"
GOLD_V2 = ROOT / "data" / "gold_holdout_v2.csv"
MMBERT_E3 = ROOT / "artifacts" / "mmbert" / "e3_base"


def read_gold(path: Path) -> pd.DataFrame:
    """Read one gold file and derive the binary target. Text stays in memory."""
    gold = pd.read_csv(path, dtype=str, keep_default_na=False)
    gold["y3"] = gold["label"].str.strip().str.lower()
    gold["y"] = (gold["y3"] != "ham").astype(int)
    return gold


def _block(gold: pd.DataFrame, p, tau: float) -> dict:
    m = metrics_at(gold["y"].to_numpy(), p, tau)
    return {
        "n": int(len(gold)),
        "n_pos": m["n_pos"],
        "n_neg": m["n_neg"],
        "recall": m["recall"],
        "fpr": m["fpr"],
        "pr_auc": m["pr_auc"],
        "roc_auc": m["roc_auc"],
        "by_language": _group_rates(gold, p, tau, "language"),
        "by_scam_type": _group_rates(gold, p, tau, "scam_type"),
    }


def score_classical(gold: pd.DataFrame, mats) -> dict:
    """Pool-arm svm and Dummy, threshold from train OOF exactly as ``score_gold``."""
    out = {}
    for name in ("svm", "dummy"):
        oof = oof_cached(mats, name)
        tau = float(threshold_for_fpr(mats.y, oof, MAX_FPR)[0])
        est = _est(name).fit(mats.X, mats.y)
        p = _score_est(est, transform_with(mats, gold["text"]))
        out[name] = {"tau": tau, "n_train": int(len(mats.y)), **_block(gold, p, tau)}
    return out


def score_mmbert(gold: pd.DataFrame, model_dir: Path, T: float, tau: float) -> dict:
    """Frozen E3 base checkpoint, temperature and threshold read from metadata."""
    from smishlens.train_mmbert import MMBertScorer

    scorer = MMBertScorer.load(model_dir, T=T)
    p = scorer.predict_proba(gold["text"].tolist())[:, 1]
    return {"tau": float(tau), "T": float(T), **_block(gold, p, tau)}


def run() -> dict:
    frame = load_pool()
    pool_train = frame[frame["split"] == "train"]
    mats = build_fold_matrices(pool_train)

    meta = json.loads((MMBERT_E3 / "metadata.json").read_text())
    T = float(meta["calibration"]["T"])
    tau = float(meta["threshold"]["value"])

    versions = {}
    for label, path in (("v1", GOLD_V1), ("v2", GOLD_V2)):
        gold = read_gold(path)
        versions[label] = {
            "path": str(path.relative_to(ROOT)),
            "n": int(len(gold)),
            "n_pos": int((gold["y"] == 1).sum()),
            "n_neg": int((gold["y"] == 0).sum()),
            "models": {
                **score_classical(gold, mats),
                "mmbert_base": score_mmbert(gold, MMBERT_E3 / "mmbert", T, tau),
            },
        }

    return {
        "synthetic": True,
        "note": (
            "second gold pass; v1 was first scored before the ExAIS ham-only "
            "change, so it is not pristine for that decision; v2 was frozen in "
            "a frozen file, identified by its SHA-256, before any model "
            "scored it; it is pristine"
        ),
        "scored_once_v2": True,
        "gold_versions": {
            "v1": {
                "pristine": False,
                "pristine_note": (
                    "first scored on 2026-11-21 before the ExAIS rebuild; this "
                    "pass is a repeat and must not be described as a first look"
                ),
                **versions["v1"],
            },
            "v2": {
                "pristine": True,
                "frozen_sha256": sha256(GOLD_V2)[:16],
                "pristine_note": "frozen before any model scored it; scored once here",
                **versions["v2"],
            },
        },
        "mmbert_base": {
            "model_dir": str((MMBERT_E3 / "mmbert").relative_to(ROOT)),
            "T": T,
            "tau": tau,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "e6_gold_v2.json"))
    args = ap.parse_args(argv)
    payload = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {out}")
    for label, block in payload["gold_versions"].items():
        parts = [
            f"{name} recall={m['recall']:.4f} fpr={m['fpr']:.4f} pr_auc={m['pr_auc']:.4f}"
            for name, m in block["models"].items()
        ]
        print(f"gold {label} n={block['n']}: " + " | ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
