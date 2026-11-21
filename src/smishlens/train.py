"""fit -> threshold -> conformal -> artifacts. The test split is NOT touched here.

The threshold comes from out-of-fold train scores under an FPR budget. The
conformal quantiles come from the untouched calibration split. Anything that
looks at test inside this module is a bug.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
from pathlib import Path

import joblib
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from smishlens.calibrate import apply_temperature, fit_temperature
from smishlens.conformal import fit_mondrian
from smishlens.data import dataset_hash
from smishlens.evaluate import threshold_for_fpr
from smishlens.features import build_final
from smishlens.normalise import CONFUSABLES
from smishlens.predict import make_manifest


def _meta(name, tau, qs, calib, train, cal, alpha, max_fpr, extra=None, seed=42) -> dict:
    return {
        "model_name": name,
        "model_version": "1.0.0",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "task": "binary: spam_or_smish vs ham",
        "datasets": {
            "sources": (
                sorted(train["source"].dropna().unique().tolist())
                if "source" in train.columns
                else []
            ),
            "dataset_sha256_16": dataset_hash(train),
            "train": len(train),
            "cal": len(cal),
            "seed": seed,
        },
        "normaliser": {
            "nfkc": True,
            "strip_format_chars": True,
            "confusables": "mixed-script",
            "n_confusables": len(CONFUSABLES),
        },
        "calibration": calib,
        "threshold": {"value": tau, "policy": f"max recall s.t. FPR <= {max_fpr}"},
        "conformal": {
            "alpha": alpha,
            "q_hat": {str(k): v for k, v in qs.items()},
            "n_cal_spam": int(cal["y"].sum()),
            "type": "class-conditional",
        },
        "policy": {
            "junk": "p>=tau and set=={1}",
            "quarantine": "set != single class",
            "deliver": "otherwise",
        },
        "explainer": {"mode": "off", "llm": None},
        "metrics_file": "reports/results/test_metrics.json",
        **(extra or {}),
    }


def _oof_scores(train, C: float, seed: int):
    """Out-of-fold probabilities, or None when the split is too small to fold."""
    counts = train["y"].value_counts()
    if len(counts) < 2 or int(counts.min()) < 5:
        return None
    return cross_val_predict(
        build_final(C),
        train["text"],
        train["y"],
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        method="predict_proba",
    )[:, 1]


def train_classical(
    train, cal, out_dir, C: float = 0.5, alpha: float = 0.05, max_fpr: float = 0.01, seed: int = 42
):
    """Fit the calibrated pipeline and freeze every decision on train/cal."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = build_final(C).fit(train["text"], train["y"])
    oof = _oof_scores(train, C, seed)
    tau = threshold_for_fpr(train["y"].values, oof, max_fpr)[0] if oof is not None else 0.5
    qs = fit_mondrian(model.predict_proba(cal["text"])[:, 1], cal["y"].values, alpha)
    joblib.dump(model, out / "model.joblib")
    meta = _meta(
        "word_char_meta_linearsvc_sigmoid",
        tau,
        qs,
        {"method": "platt(sigmoid, cv=5)"},
        train,
        cal,
        alpha,
        max_fpr,
        seed=seed,
    )
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    make_manifest(out)
    return out


def train_mmbert_pipeline(
    train, cal, out_dir, alpha: float = 0.05, max_fpr: float = 0.01, seed: int = 42, **kw
):
    """No out-of-fold pass for a transformer, so the cal split is cut in two.

    ``cal_a`` fits temperature and tau, ``cal_b`` fits conformal. Each half still
    needs hundreds of spam rows; check before trusting the quantile.
    """
    from smishlens import train_mmbert as tm

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tm.train(train, str(out / "mmbert"), seed=seed, **kw)
    cal_a, cal_b = train_test_split(cal, test_size=0.5, stratify=cal["y"], random_state=seed)
    ma = tm.cal_margins(str(out / "mmbert"), cal_a["text"])
    mb = tm.cal_margins(str(out / "mmbert"), cal_b["text"])
    T = fit_temperature(ma, cal_a["y"].values)
    tau = threshold_for_fpr(cal_a["y"].values, apply_temperature(ma, T), max_fpr)[0]
    qs = fit_mondrian(apply_temperature(mb, T), cal_b["y"].values, alpha)
    meta = _meta(
        "mmbert_base_temp_scaled",
        tau,
        qs,
        {"method": "temperature", "T": T},
        train,
        cal,
        alpha,
        max_fpr,
        {"torch_note": "record torch/transformers versions here"},
        seed=seed,
    )
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    make_manifest(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the classical detector on the UCI anchor.")
    ap.add_argument("--data", default="data/raw/uci/SMSSpamCollection")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-fpr", type=float, default=0.01)
    args = ap.parse_args(argv)

    from smishlens.data import clean_dedup, load_uci, split_pool

    frame = clean_dedup(load_uci(args.data))
    train, cal, test = split_pool(frame)
    out = train_classical(train, cal, args.out, C=args.C, alpha=args.alpha, max_fpr=args.max_fpr)
    print(f"train={len(train)} cal={len(cal)} test={len(test)} -> {out}")
    return out


if __name__ == "__main__":
    main()
