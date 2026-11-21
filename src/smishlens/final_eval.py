"""The ONE evaluation of the frozen test split.

Run after ``python -m smishlens.train``. Refuses to overwrite an existing
``test_metrics.json`` unless ``--force`` is passed: the test split is touched
exactly once, and re-running it after seeing the numbers is how test-set
overfitting starts.

Usage: python -m smishlens.final_eval [--artifacts artifacts] [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smishlens.conformal import conformal_set, coverage
from smishlens.evaluate import boot_ci, metrics_at, ppv_table
from smishlens.policy import decide
from smishlens.predict import load_model


def _bootstrap(y, p, tau: float) -> dict[str, dict[str, float]]:
    return {
        key: dict(zip(("lo", "hi"), boot_ci(y, p, tau, key=key, n=1000, seed=42), strict=True))
        for key in ("recall", "precision", "f1", "pr_auc")
    }


def evaluate(artifacts: str, test_csv: str, out_path: Path) -> dict:
    bundle = load_model(artifacts)
    meta = bundle.meta
    test = pd.read_csv(test_csv, encoding_errors="replace")
    y = test["y"].astype(int).to_numpy()
    p = bundle.model.predict_proba(test["text"].astype(str).tolist())[:, 1]
    tau = float(meta["threshold"]["value"])
    qs = {int(k): float(v) for k, v in meta["conformal"]["q_hat"].items()}

    point = metrics_at(y, p, tau)
    sets = [conformal_set(float(s), qs) for s in p]
    cov, abstain = coverage(p, y, qs)
    actions = [decide(float(s), tau, S) for s, S in zip(p, sets, strict=True)]
    action_counts = {a: actions.count(a) for a in ("deliver", "quarantine", "junk")}

    junk = np.array([a == "junk" for a in actions])
    pos, neg = int(y.sum()), int((y == 0).sum())
    tp = int((junk & (y == 1)).sum())
    fp = int((junk & (y == 0)).sum())

    payload = {
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "split": "test (frozen, touched once)",
        "test_file": str(test_csv),
        "n_rows": len(test),
        "n_pos": pos,
        "n_neg": neg,
        "prevalence": float(y.mean()),
        "model_name": meta.get("model_name"),
        "model_version": meta.get("model_version"),
        "dataset_sha256_16": meta.get("datasets", {}).get("dataset_sha256_16"),
        "threshold": {"value": tau, "policy": meta.get("threshold", {}).get("policy")},
        "conformal": {"alpha": meta.get("conformal", {}).get("alpha"), "q_hat": qs},
        "operating_point": point,
        "bootstrap_95ci": _bootstrap(y, p, tau),
        "conformal_coverage": {str(k): v for k, v in cov.items()},
        "abstain_rate": abstain,
        "policy_actions": action_counts,
        "policy_junk_confusion": {"tp": tp, "fp": fp, "fn": pos - tp, "tn": neg - fp},
        "ppv_by_prevalence": ppv_table(point["recall"], point["fpr"]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate the frozen UCI test split exactly once.")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--test", default="data/processed/uci/test.csv")
    ap.add_argument("--out", default="reports/results/test_metrics.json")
    ap.add_argument("--force", action="store_true", help="overwrite an existing metrics file")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; the test split was already evaluated. Use --force only deliberately.")

    payload = evaluate(args.artifacts, args.test, out)
    op = payload["operating_point"]
    print(f"test n={payload['n_rows']} prevalence={payload['prevalence']:.3f}")
    print(
        f"  recall={op['recall']:.4f} fpr={op['fpr']:.4f} "
        f"precision={op['precision']:.4f} f1={op['f1']:.4f} "
        f"pr_auc={op['pr_auc']:.4f} roc_auc={op['roc_auc']:.4f}"
    )
    print(f"  actions={payload['policy_actions']} abstain={payload['abstain_rate']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
