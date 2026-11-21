"""Metrics and the cost-based second operating point."""

import math

import numpy as np
import pytest

from smishlens.evaluate import cost_operating_point, metrics_at, ppv


@pytest.fixture
def overlapping():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.7, 0.5, 0.3, 0.6, 0.4, 0.2, 0.1])
    return y, p


def test_cost_point_avoids_expensive_false_positives(overlapping):
    y, p = overlapping
    point = cost_operating_point(y, p, fp_cost=10.0, fn_cost=1.0)
    # One false positive costs ten misses, so the optimum gives up two positives.
    assert point["tau"] > 0.6
    assert point["recall"] == pytest.approx(0.5)
    assert point["fpr"] == pytest.approx(0.0)
    assert point["cost"] == pytest.approx(2.0)


def test_cost_point_moves_when_false_positives_are_cheap(overlapping):
    y, p = overlapping
    point = cost_operating_point(y, p, fp_cost=1.0, fn_cost=1.0)
    assert point["tau"] < 0.6
    assert point["recall"] == pytest.approx(1.0)
    assert point["fpr"] == pytest.approx(0.5)
    assert point["cost"] == pytest.approx(2.0)


def test_cost_point_beats_a_fixed_threshold(overlapping):
    y, p = overlapping
    point = cost_operating_point(y, p, fp_cost=10.0, fn_cost=1.0)
    fixed = metrics_at(y, p, 0.5)
    fixed_cost = 10.0 * fixed["fpr"] * fixed["n_neg"] + (1 - fixed["recall"]) * fixed["n_pos"]
    assert point["cost"] <= fixed_cost


def test_cost_point_empty():
    point = cost_operating_point([], [])
    assert math.isnan(point["tau"]) and math.isnan(point["cost"])


def test_ppv_matches_worked_example():
    # Roadmap §7.6: TPR 98.66%, FPR 0.82% at 5% prevalence -> 86.36%.
    assert ppv(0.9866, 0.0082, 0.05) == pytest.approx(0.8636, abs=1e-3)


def test_cost_point_handles_tied_scores_as_one_block():
    # Two negatives share 0.6, so no threshold can split them.
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.6, 0.6, 0.2])
    point = cost_operating_point(y, p, fp_cost=10.0, fn_cost=1.0)
    # Keeping the tie block costs 20 and saves one miss, so drop both positives.
    assert point["tau"] == pytest.approx(0.9)
    assert point["recall"] == pytest.approx(0.5)
    assert point["fpr"] == pytest.approx(0.0)
    assert point["cost"] == pytest.approx(1.0)


def test_cost_point_matches_brute_force_on_random_scores():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(6, 40))
        y = rng.integers(0, 2, n)
        p = np.round(rng.random(n), 1)
        point = cost_operating_point(y, p, fp_cost=10.0, fn_cost=1.0)
        candidates = [float("inf"), *np.unique(p)]
        best = min(
            10.0 * int(((p >= tau) & (y == 0)).sum())
            + 1.0 * int(((p < tau) & (y == 1)).sum())
            for tau in candidates
        )
        assert point["cost"] == pytest.approx(best)
        assert point["cost"] == pytest.approx(
            10.0 * int(((p >= point["tau"]) & (y == 0)).sum())
            + 1.0 * int(((p < point["tau"]) & (y == 1)).sum())
        )
