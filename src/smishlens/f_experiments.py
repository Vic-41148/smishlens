"""F1-F5: svm-only fixes under LOSO, no gold.

Folds are the both-class sources ``mishra`` and ``moz`` plus the ham-only
``exais`` fold. The macro is the unweighted mean over mishra and moz only;
ExAIS FPR is reported separately and never enters the macro.

Every arm gets its own train-OOF 1% FPR threshold. The baseline arm runs
through this module's own path and is checked against ``e4_loso.json`` before
any other arm runs.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from smishlens.evaluate import metrics_at, threshold_for_fpr
from smishlens.features import build_features
from smishlens.ham_category import ORDER, classify
from smishlens.normalise import EMAIL, MONEY, PHONE, SHORT, URL
from smishlens.pool_experiments import (
    MAX_FPR,
    RESULTS,
    _fold,
    load_pool,
    run_e5,
)

SEED = 42
FOLDS = ("mishra", "moz", "exais")
BOTH = ("mishra", "moz")
E4_JSON = RESULTS / "e4_loso.json"
E5_JSON = RESULTS / "e5_shortcut.json"
TIME_BOX_S = 15 * 60
SUCCESS_RULE = (
    "macro LOSO FPR at the arm's own frozen 1% threshold must fall by at least "
    "0.05 versus the baseline, while macro LOSO PR-AUC may fall by at most 0.02. "
    "Every arm is reported, including failures."
)


def format_masked(text: object) -> str:
    """Digits->0, currency/phone/URL masks, case-fold. No other change."""
    s = str(text)
    s = EMAIL.sub(" <EMAIL> ", s)
    s = URL.sub(" <URL> ", s)
    s = MONEY.sub(" <MONEY> ", s)
    s = PHONE.sub(" <PHONE> ", s)
    s = SHORT.sub(" <SHORTCODE> ", s)
    s = re.sub(r"\d", "0", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _pipe(class_weight, feats_kw: dict | None):
    return Pipeline(
        [
            ("feats", build_features(**(feats_kw or {}))),
            (
                "clf",
                LinearSVC(
                    C=0.5,
                    class_weight=class_weight,
                    dual="auto",
                    random_state=SEED,
                ),
            ),
        ]
    )


def _evaluate_arm(
    train,
    test,
    *,
    weights: np.ndarray | None = None,
    class_weight: str | None = "balanced",
    feats_kw: dict | None = None,
) -> dict:
    """Fit once, freeze the 1% FPR threshold on train OOF, score test once."""
    pipe = _pipe(class_weight, feats_kw)
    params = {} if weights is None else {"clf__sample_weight": weights}
    oof = cross_val_predict(
        pipe,
        train["text"],
        train["y"],
        method="decision_function",
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        params=params,
        n_jobs=1,
    )
    tau = float(threshold_for_fpr(train["y"].to_numpy(), oof, MAX_FPR)[0])
    pipe.fit(train["text"], train["y"], **params)
    p = pipe.decision_function(test["text"])
    return {"tau": tau, "p": p, "oof": oof}


def _prevalence_weights(target: float):
    """Class weights that make the effective training prevalence ``target``."""

    def fn(train):
        y = train["y"].to_numpy()
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        w = np.ones(len(y), dtype=float)
        w[y == 1] = (target / (1.0 - target)) * (n_neg / n_pos)
        return w

    return fn


def _source_class_weights(train):
    """Equal total weight per (source, class) cell."""
    counts = train.groupby(["source", "y"])["text"].transform("size").to_numpy()
    return 1.0 / counts


def _run_arm(
    frame,
    *,
    weights_fn=None,
    class_weight: str | None = "balanced",
    feats_kw: dict | None = None,
    text_fn=None,
    folds=FOLDS,
) -> tuple[dict, dict]:
    start = time.perf_counter()
    per_source = {}
    raw = {}
    for held in folds:
        train, test = _fold(frame, held)
        if text_fn is not None:
            train = train.assign(text=train["text"].map(text_fn))
            test = test.assign(text=test["text"].map(text_fn))
        weights = weights_fn(train) if weights_fn is not None else None
        res = _evaluate_arm(
            train,
            test,
            weights=weights,
            class_weight=class_weight,
            feats_kw=feats_kw,
        )
        test = test.reset_index(drop=True)
        m = metrics_at(test["y"].to_numpy(), res["p"], res["tau"])
        row = {
            "mode": "fpr_only" if held == "exais" else "both",
            "tau": res["tau"],
            "n_pos": m["n_pos"],
            "n_neg": m["n_neg"],
        }
        if held == "exais":
            row["fpr"] = m["fpr"]
        else:
            row["recall"] = m["recall"]
            row["fpr"] = m["fpr"]
            row["pr_auc"] = m["pr_auc"]
        per_source[held] = row
        raw[held] = {
            "p": res["p"],
            "tau": res["tau"],
            "oof": res["oof"],
            "train_y": train["y"].to_numpy(),
            "test": test,
        }

    macro = {
        f"macro_{key}": float(np.mean([per_source[src][key] for src in BOTH]))
        for key in ("recall", "fpr", "pr_auc")
    }
    return (
        {
            "per_source": per_source,
            "macro": macro,
            "elapsed_s": time.perf_counter() - start,
        },
        raw,
    )


def _e4_reference() -> dict:
    data = json.loads(E4_JSON.read_text())
    rows = [r for r in data["rows"] if r.get("model") == "svm"]
    macro = next(r for r in rows if r.get("source") == "macro")
    per = {r["source"]: r for r in rows if r.get("source") in FOLDS}
    return {"macro": macro, "per_source": per}


def _guard(baseline: dict, ref: dict) -> dict:
    observed = baseline["macro"]
    expected = ref["macro"]
    for key in ("macro_recall", "macro_fpr", "macro_pr_auc"):
        if abs(observed[key] - expected[key]) > 1e-6:
            raise SystemExit(
                f"guard failed: {key} {observed[key]:.12f} vs {expected[key]:.12f}"
            )
    return {
        "expected": {k: expected[k] for k in ("macro_recall", "macro_fpr", "macro_pr_auc")},
        "observed": observed,
        "tolerance": 1e-6,
        "passed": True,
    }


def _f1(raw: dict) -> dict:
    out = {}
    for held in FOLDS:
        test = raw[held]["test"]
        p = np.asarray(raw[held]["p"], dtype=float)
        tau = raw[held]["tau"]
        y = test["y"].to_numpy()
        ham = test[y == 0].copy()
        hp = p[y == 0]
        ham["category"] = ham["text"].map(classify)
        block = {}
        for cat in ORDER:
            mask = (ham["category"] == cat).to_numpy()
            n = int(mask.sum())
            flagged = int((hp[mask] >= tau).sum())
            block[cat] = {
                "n_ham": n,
                "n_flagged": flagged,
                "fpr": (flagged / n) if n else None,
            }
        block["total"] = {
            "n_ham": int(len(ham)),
            "n_flagged": int((hp >= tau).sum()),
            "fpr": float((hp >= tau).mean()) if len(ham) else None,
        }
        out[held] = block
    return out


def _logit(q):
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(q / (1 - q))


def _em_prior(p, init: float, tol: float = 1e-10, max_iter: int = 1000) -> float:
    """Saerens-Latinne-Decaestecker prior estimate under p(x|y) fixed."""
    pi = float(init)
    for _ in range(max_iter):
        w = pi * p / np.clip(pi * p + (1 - pi) * (1 - p), 1e-12, None)
        new = float(w.mean())
        if abs(new - pi) < tol:
            return new
        pi = new
    return pi


def _adjust_prior(p, source_prior: float, target_prior: float):
    return 1.0 / (
        1.0
        + np.exp(-(_logit(p) + _logit(target_prior) - _logit(source_prior)))
    )


def _f5(raw: dict) -> dict:
    out = {}
    for held in FOLDS:
        test = raw[held]["test"]
        y = test["y"].to_numpy()
        p = np.asarray(raw[held]["p"], dtype=float)
        oof = np.asarray(raw[held]["oof"], dtype=float)
        y_train = np.asarray(raw[held]["train_y"], dtype=int)

        platt = LogisticRegression(C=1e6, max_iter=1000).fit(
            oof.reshape(-1, 1), y_train
        )
        p_cal = platt.predict_proba(p.reshape(-1, 1))[:, 1]
        p_cal_oof = platt.predict_proba(oof.reshape(-1, 1))[:, 1]
        source_prior = float(y_train.mean())
        true_prior = float(y.mean())
        em_prior = _em_prior(p_cal, source_prior)

        p_adj = _adjust_prior(p_cal, source_prior, em_prior)
        p_adj_oof = _adjust_prior(p_cal_oof, source_prior, em_prior)
        tau_adj = float(threshold_for_fpr(y_train, p_adj_oof, MAX_FPR)[0])
        frozen = metrics_at(y, p, raw[held]["tau"])
        adjusted = metrics_at(y, p_adj, tau_adj)
        out[held] = {
            "true_prevalence": true_prior,
            "source_prevalence": source_prior,
            "em_prevalence": em_prior,
            "frozen": {
                "tau": raw[held]["tau"],
                "recall": frozen["recall"],
                "fpr": frozen["fpr"],
            },
            "prior_adjusted": {
                "tau": tau_adj,
                "recall": adjusted["recall"],
                "fpr": adjusted["fpr"],
            },
        }
    return out


def _verdict(arm: dict, baseline: dict) -> dict:
    delta_fpr = arm["macro"]["macro_fpr"] - baseline["macro"]["macro_fpr"]
    delta_pr = arm["macro"]["macro_pr_auc"] - baseline["macro"]["macro_pr_auc"]
    return {
        "delta_macro_fpr": float(delta_fpr),
        "delta_macro_pr_auc": float(delta_pr),
        "success": bool(delta_fpr <= -0.05 and delta_pr >= -0.02),
    }


def run(guard_only: bool = False) -> dict:
    frame = load_pool()
    ref = _e4_reference()

    baseline, baseline_raw = _run_arm(frame)
    guard = _guard(baseline, ref)
    payload = {
        "note": (
            "svm only; no gold; folds mishra, moz, exais; macro is the "
            "unweighted mean over mishra and moz; ExAIS FPR is separate and "
            "never enters the macro"
        ),
        "scope": "LOSO, each arm gets its own train-OOF 1% FPR threshold",
        "success_rule": SUCCESS_RULE,
        "guard": guard,
        "arms": {"baseline": baseline},
    }
    if guard_only:
        return payload

    arms = {
        "prevalence_50": dict(
            weights_fn=_prevalence_weights(0.50), class_weight=None
        ),
        "prevalence_20": dict(
            weights_fn=_prevalence_weights(0.20), class_weight=None
        ),
        "prevalence_05": dict(
            weights_fn=_prevalence_weights(0.05), class_weight=None
        ),
        "source_class_balanced": dict(
            weights_fn=_source_class_weights, class_weight=None
        ),
        "format_masked": dict(
            feats_kw={"normalise_on": False}, text_fn=format_masked
        ),
    }

    time_box = {"limit_s": TIME_BOX_S, "violated": False, "arms": {}}
    for name, kwargs in arms.items():
        arm, _ = _run_arm(frame, **kwargs)
        arm["verdict"] = _verdict(arm, baseline)
        payload["arms"][name] = arm
        time_box["arms"][name] = arm["elapsed_s"]
        if arm["elapsed_s"] > TIME_BOX_S:
            time_box["violated"] = True

    payload["time_box"] = time_box
    payload["f1_fp_by_ham_category"] = _f1(baseline_raw)
    payload["f5_prior_em"] = _f5(baseline_raw)

    masked = frame.copy()
    masked["text"] = masked["text"].map(format_masked)
    e5_masked = run_e5(masked)
    payload["f4_e5"] = {
        "baseline": {
            key: json.loads(E5_JSON.read_text())[key]
            for key in (
                "imc25_vs_rest_auc_from_format",
                "exais_vs_rest_auc_from_format",
                "format_only_auc_pooled",
            )
        },
        "format_masked": {
            key: e5_masked[key]
            for key in (
                "imc25_vs_rest_auc_from_format",
                "exais_vs_rest_auc_from_format",
                "format_only_auc_pooled",
            )
        },
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "f_experiments.json"))
    ap.add_argument("--guard-only", action="store_true")
    args = ap.parse_args(argv)

    payload = run(guard_only=args.guard_only)
    if args.guard_only:
        print(json.dumps(payload["guard"], indent=2))
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"guard {payload['guard']['observed']}")
    for name, arm in payload["arms"].items():
        v = arm.get("verdict", {})
        print(
            f"{name}: macro recall {arm['macro']['macro_recall']:.4f} "
            f"fpr {arm['macro']['macro_fpr']:.4f} "
            f"pr_auc {arm['macro']['macro_pr_auc']:.4f} "
            f"exais_fpr {arm['per_source']['exais']['fpr']:.4f} "
            f"success={v.get('success')} elapsed={arm['elapsed_s']:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
