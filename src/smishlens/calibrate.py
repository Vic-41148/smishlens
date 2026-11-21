"""Temperature scaling for the encoder, plus reliability metrics.

Classical models are calibrated by Platt scaling inside the pipeline. A
fine-tuned encoder is not, so it gets one scalar fitted on the calibration
split, never on test.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def fit_temperature(margins, y) -> float:
    """Fit ``T`` minimising log-loss of ``sigmoid(margin / T)``.

    ``margins`` is ``logit_spam - logit_ham`` on the calibration split. One
    parameter, so it cannot change the ranking, only the confidence.
    """
    m = np.asarray(margins, dtype=float)
    y = np.asarray(y, dtype=float)

    def nll(log_t: float) -> float:
        p = np.clip(_sigmoid(m / np.exp(log_t)), 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def apply_temperature(margins, T: float):
    """Turn margins into calibrated probabilities with the fitted ``T``."""
    return _sigmoid(np.asarray(margins, dtype=float) / float(T))


def fit_platt(scores, y) -> tuple[float, float]:
    """Fit Platt ``sigmoid(a * score + b)`` on raw decision scores.

    Used only where the estimator exposes ``decision_function`` and no
    probability, such as the pool ``LinearSVC`` arm. Fit on the calibration
    half that is not used for evaluation.
    """
    from sklearn.linear_model import LogisticRegression

    z = np.asarray(scores, dtype=float).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z, np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(scores, a: float, b: float):
    """Apply a fitted Platt map to raw decision scores."""
    return _sigmoid(float(a) * np.asarray(scores, dtype=float) + float(b))


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error, equal-width bins."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def brier(p, y) -> float:
    """Mean squared error of the probability, lower is better."""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y)) ** 2))


def reliability(p, y, bins: int = 10) -> list[dict[str, float]]:
    """Rows for a reliability diagram: mean confidence vs observed frequency."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            rows.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": int(mask.sum()),
                    "confidence": float(p[mask].mean()),
                    "frequency": float(y[mask].mean()),
                }
            )
    return rows
