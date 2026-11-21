"""E-recal: recalibrate tau on k labelled target-domain ham, under LOSO.

This is a descriptive stress test, not a zero-shot result. It uses labelled ham
from the held-out source to set the threshold, then evaluates on the remaining
held-out rows. The frozen LOSO threshold is the comparison point.

mmBERT-small is re-scored from the frozen E4 checkpoints with no refit. The
classical svm has no saved fold checkpoint, so its fold pipeline is refit with
the identical config and seed as E4; that is a deterministic repeat, disclosed
here and in the log. No held-out row is used for model fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from smishlens.evaluate import metrics_at
from smishlens.pool_experiments import RESULTS, ROOT

RESULTS_DIR = RESULTS
E4_CLASSICAL = RESULTS_DIR / "e4_loso.json"
E4_MMBERT = RESULTS_DIR / "e4_loso_mmbert_small.json"
E4_MMBERT_DIR = ROOT / "artifacts" / "mmbert" / "e4_small"
K_VALUES = (25, 50, 100)
RESPLITS = 20
SEED = 42
MAX_FPR = 0.01


def _frozen_row(path: Path, model: str, source: str) -> dict:
    data = json.loads(path.read_text())
    for row in data["rows"]:
        if row.get("model") == model and row.get("source") == source:
            return row
    raise KeyError(f"{model}/{source} not in {path.name}")


def _recal(p: np.ndarray, y: np.ndarray, k: int, rng: np.random.Generator) -> dict:
    ham = np.flatnonzero(y == 0)
    draws = []
    for _ in range(RESPLITS):
        pick = rng.choice(ham, size=k, replace=False)
        tau = float(np.quantile(p[pick], 1 - MAX_FPR))
        keep = np.ones(len(y), dtype=bool)
        keep[pick] = False
        m = metrics_at(y[keep], p[keep], tau)
        cal_fpr = float(np.mean(p[pick] >= tau))
        draws.append({"recall": m["recall"], "fpr": m["fpr"], "cal_fpr": cal_fpr})

    def stat(key: str) -> dict:
        values = np.asarray([d[key] for d in draws], dtype=float)
        return {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return {
        "k": k,
        "n_eval": int(len(y) - k),
        "calibration_fpr": stat("cal_fpr"),
        "recall": stat("recall"),
        "fpr": stat("fpr"),
    }


def run() -> dict:
    from smishlens.pool_experiments import _fold, evaluate, load_pool
    from smishlens.train_mmbert import MMBertScorer

    frame = load_pool()
    rng = np.random.default_rng(SEED)
    out = {
        "note": (
            "uses labelled target-domain ham to set tau; not zero-shot. svm fold "
            "pipeline refit with the E4 config/seed because no checkpoint was "
            "saved; mmBERT-small re-scored from frozen E4 checkpoints, no refit"
        ),
        "k_values": list(K_VALUES),
        "resplits": RESPLITS,
        "seed": SEED,
        "models": {"svm": {}, "mmbert_small": {}},
    }

    for held in ("mishra", "moz"):
        train, test = _fold(frame, held)
        y = test["y"].to_numpy()

        res = evaluate(train, test, "svm")
        p_svm = np.asarray(res["p"], dtype=float)
        tau_svm = float(res["tau"])
        frozen_svm = _frozen_row(E4_CLASSICAL, "svm", held)
        assert abs(tau_svm - frozen_svm["tau"]) < 1e-9, (held, tau_svm, frozen_svm["tau"])

        fold_dir = E4_MMBERT_DIR / f"heldout_{held}"
        meta = json.loads((fold_dir / "metadata.json").read_text())
        T = float(meta["calibration"]["T"])
        tau_mm = float(meta["threshold"]["value"])
        scorer = MMBertScorer.load(fold_dir / "mmbert", T=T)
        p_mm = np.asarray(scorer.predict_proba(test["text"].astype(str).tolist())[:, 1])
        frozen_mm = _frozen_row(E4_MMBERT, "mmbert_small", held)
        assert abs(tau_mm - frozen_mm["tau"]) < 1e-9

        for name, p, tau in (
            ("svm", p_svm, tau_svm),
            ("mmbert_small", p_mm, tau_mm),
        ):
            m = metrics_at(y, p, tau)
            out["models"][name][held] = {
                "n_pos": int((y == 1).sum()),
                "n_neg": int((y == 0).sum()),
                "frozen_tau": tau,
                "frozen": {"recall": m["recall"], "fpr": m["fpr"]},
                "recal": {str(k): _recal(p, y, k, rng) for k in K_VALUES},
            }

    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS_DIR / "e_recal.json"))
    args = ap.parse_args(argv)
    payload = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    for name, sources in payload["models"].items():
        for held, block in sources.items():
            print(
                f"{name} {held}: frozen recall {block['frozen']['recall']:.3f} "
                f"fpr {block['frozen']['fpr']:.3f}"
            )
            for k, r in block["recal"].items():
                print(
                    f"  k={k}: recall {r['recall']['mean']:.3f} "
                    f"fpr {r['fpr']['mean']:.3f} cal_fpr {r['calibration_fpr']['mean']:.3f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
