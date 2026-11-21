"""Class-conditional (Mondrian) conformal abstention.

Fit on the calibration split ONLY. The guarantee is marginal per class and
needs exchangeability, which a held-out source does not give you; that gap is
the point of the LOSO conformal experiment (RQ5).
"""

from __future__ import annotations

import numpy as np


def fit_mondrian(p_cal, y_cal, alpha: float = 0.05) -> dict[int, float]:
    """Per-class quantile of the nonconformity score ``1 - P(true class)``.

    Returns ``{0: q_ham, 1: q_spam}``. A class with too few calibration points
    gets ``1.0``, which keeps it in every prediction set: abstain rather than
    guess from a handful of examples.
    """
    p_cal = np.asarray(p_cal, dtype=float)
    y_cal = np.asarray(y_cal)
    qs: dict[int, float] = {}
    for c in (0, 1):
        s = 1 - p_cal if c == 1 else p_cal
        scores = np.sort(s[y_cal == c])
        n = len(scores)
        k = int(np.ceil((n + 1) * (1 - alpha)))
        qs[c] = float(scores[k - 1]) if 0 < k <= n else 1.0
    return qs


def conformal_set(p_spam: float, qs: dict[int, float]) -> list[int]:
    """Prediction set at one calibrated probability.

    ``[0]`` ham, ``[1]`` spam, ``[0, 1]`` ambiguous, ``[]`` unlike either class.
    """
    out: list[int] = []
    if p_spam <= qs[0]:
        out.append(0)
    if 1 - p_spam <= qs[1]:
        out.append(1)
    return out


def coverage(p, y, qs: dict[int, float]) -> tuple[dict[int, float], float]:
    """Per-class empirical coverage and the abstain rate (set size != 1)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    sets = [conformal_set(x, qs) for x in p]
    cov = {
        int(c): float(np.mean([c in s for s, t in zip(sets, y, strict=True) if t == c]))
        for c in (0, 1)
        if (y == c).any()
    }
    abstain = float(np.mean([len(s) != 1 for s in sets]))
    return cov, abstain


def resplit_coverage(
    p,
    y,
    alpha: float = 0.05,
    n_splits: int = 20,
    cal_frac: float = 0.5,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Coverage over >= 20 random re-splits.

    One draw is not a result: with a small spam calibration set the spam
    coverage can swing from 0.88 to 1.00 between draws.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    out: list[dict[str, float]] = []
    for _ in range(n_splits):
        idx = rng.permutation(len(y))
        k = int(len(y) * cal_frac)
        cal, ev = idx[:k], idx[k:]
        cov, abstain = coverage(p[ev], y[ev], fit_mondrian(p[cal], y[cal], alpha))
        out.append(
            {
                "cov_ham": cov.get(0, float("nan")),
                "cov_spam": cov.get(1, float("nan")),
                "abstain": abstain,
            }
        )
    return out
