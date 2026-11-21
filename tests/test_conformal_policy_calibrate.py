"""Conformal sets, the policy layer and temperature scaling."""

import numpy as np
import pytest

from smishlens.calibrate import apply_temperature, brier, ece, fit_temperature, reliability
from smishlens.conformal import conformal_set, coverage, fit_mondrian, resplit_coverage
from smishlens.policy import decide, decide_ablation


@pytest.mark.parametrize(
    "p, S, tau, expected",
    [
        (0.95, {1}, 0.90, "junk"),
        (0.50, {0, 1}, 0.90, "quarantine"),
        (0.95, set(), 0.90, "quarantine"),
        (0.10, {0}, 0.90, "deliver"),
        (0.50, {1}, 0.90, "deliver"),
    ],
)
def test_policy_table(p, S, tau, expected):
    assert decide(p, tau, S) == expected


def test_policy_ablation_quarantines_disagreement():
    assert decide_ablation(0.50, 0.90, {1}) == "quarantine"


def test_mondrian_perfect_scores():
    p = np.r_[np.full(100, 0.1), np.full(100, 0.9)]
    y = np.r_[np.zeros(100, int), np.ones(100, int)]
    qs = fit_mondrian(p, y, 0.05)
    assert qs[0] == pytest.approx(0.1)
    assert qs[1] == pytest.approx(0.1)
    assert conformal_set(0.9, qs) == [1]
    assert conformal_set(0.05, qs) == [0]
    assert conformal_set(0.5, qs) == []


def test_coverage_perfect():
    p = np.r_[np.full(50, 0.1), np.full(50, 0.9)]
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    qs = fit_mondrian(p, y, 0.05)
    cov, abstain = coverage(p, y, qs)
    assert cov[0] == 1.0
    assert cov[1] == 1.0
    assert abstain == 0.0


def test_resplit_coverage_length():
    rng = np.random.default_rng(0)
    p = rng.random(200)
    y = (p > 0.5).astype(int)
    rows = resplit_coverage(p, y, n_splits=5)
    assert len(rows) == 5
    assert all({"cov_ham", "cov_spam", "abstain"} <= set(row) for row in rows)


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    margins = rng.normal(0, 5, 1000)
    raw = 1 / (1 + np.exp(-margins))
    T = fit_temperature(margins, y)
    scaled = apply_temperature(margins, T)
    assert T > 1
    assert ece(scaled, y) <= ece(raw, y) + 1e-6


def test_brier_and_reliability():
    assert brier([0, 1], [0, 1]) == 0.0
    rows = reliability([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], bins=2)
    assert len(rows) == 2
