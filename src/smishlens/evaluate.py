"""Metrics, operating point, prevalence-aware precision, intervals, paired test."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def ppv(tpr: float, fpr: float, prevalence: float) -> float:
    """Positive predictive value at a given prevalence."""
    denominator = tpr * prevalence + fpr * (1 - prevalence)
    return float(tpr * prevalence / denominator) if denominator else float("nan")


def ppv_table(tpr: float, fpr: float, prevalences=(0.5, 0.2, 0.1, 0.05), per: int = 1000):
    """Precision and error counts per 1000 messages at several spam rates."""
    return [
        {
            "prevalence": prev,
            "ppv": ppv(tpr, fpr, prev),
            "caught_per_1000": tpr * prev * per,
            "false_alarms_per_1000": fpr * (1 - prev) * per,
        }
        for prev in prevalences
    ]


def threshold_for_fpr(y, p, max_fpr: float = 0.01):
    """Best TPR within the FPR budget.

    Call on OUT-OF-FOLD or calibration scores, never on test.
    Returns (threshold, tpr, fpr).

    When not even the second-highest score is inside the budget, the only point
    that qualifies is "predict nothing". That is returned as a finite sentinel
    above every observed score rather than ``inf``, so the value survives JSON
    and still means the same thing for probabilities and decision functions.
    """
    p = np.asarray(p)
    fpr, tpr, thr = roc_curve(y, p)
    within = np.where(fpr <= max_fpr)[0]
    i = int(within[-1])
    tau = float(thr[i])
    if not np.isfinite(tau):
        tau = float(p.max()) + 1.0 if p.size else 1.0
    return tau, float(tpr[i]), float(fpr[i])


def metrics_at(y, p, tau: float) -> dict[str, float]:
    """Confusion-matrix metrics at one threshold."""
    y = np.asarray(y)
    p = np.asarray(p)
    pred = p >= tau
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    recall = tp / pos if pos else float("nan")
    fpr = fp / neg if neg else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall)
        else float("nan")
    )
    both = pos > 0 and neg > 0
    return {
        "n_pos": pos,
        "n_neg": neg,
        "recall": recall,
        "fpr": fpr,
        "precision": precision,
        "f1": f1,
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
    }


def boot_ci(y, p, tau: float, key: str = "recall", n: int = 1000, seed: int = 0, level: float = 0.95):
    """Bootstrap percentile interval for one metric."""
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        value = metrics_at(y[idx], p[idx], tau)[key]
        if value == value:  # skip NaN
            values.append(value)
    lo, hi = np.percentile(values, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def mcnemar_exact(y, pred_a, pred_b):
    """Exact McNemar on paired correctness. Returns (b, c, p)."""
    y = np.asarray(y)
    correct_a = np.asarray(pred_a) == y
    correct_b = np.asarray(pred_b) == y
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(min(b, c), b + c, 0.5).pvalue)


def cost_operating_point(
    y, p, fp_cost: float = 10.0, fn_cost: float = 1.0
) -> dict[str, float]:
    """Threshold minimising expected cost when a false positive costs ``fp_cost``.

    The FPR budget is the headline operating point; this is the second one,
    because a blocked OTP or bank alert costs far more than a missed ad. Every
    candidate threshold is tried, so the result is the exact minimiser on the
    observed scores, not a grid approximation.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    if y.size == 0:
        return {"tau": float("nan"), "cost": float("nan"), "recall": float("nan"), "fpr": float("nan")}
    # Same exact minimiser as scoring every observed threshold, but in O(n log n):
    # sweep the unique scores upwards and keep suffix counts of positives and
    # negatives at or above each one. Ties move as a block, as in roc_curve.
    scores, inverse = np.unique(p, return_inverse=True)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pos = np.bincount(inverse, weights=(y == 1).astype(float), minlength=scores.size)
    neg = np.bincount(inverse, weights=(y == 0).astype(float), minlength=scores.size)
    pos_ge = np.cumsum(pos[::-1])[::-1]
    neg_ge = np.cumsum(neg[::-1])[::-1]

    costs = fp_cost * neg_ge + fn_cost * (n_pos - pos_ge)
    k = int(np.argmin(costs))  # first minimum in ascending score order
    best = {
        "tau": float(scores[k]),
        "cost": float(costs[k]),
        "recall": float(pos_ge[k] / n_pos) if n_pos else float("nan"),
        "fpr": float(neg_ge[k] / n_neg) if n_neg else float("nan"),
    }
    # Predicting nothing is always a candidate; it wins only on a strict tie-break.
    if fn_cost * n_pos < best["cost"]:
        best = {
            "tau": float("inf"),
            "cost": float(fn_cost * n_pos),
            "recall": 0.0 if n_pos else float("nan"),
            "fpr": 0.0 if n_neg else float("nan"),
        }
    return best
