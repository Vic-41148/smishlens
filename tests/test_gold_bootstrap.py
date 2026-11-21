"""Bootstrap helpers on synthetic gold-sized arrays (no message text)."""

from __future__ import annotations

import numpy as np

import smishlens.gold_bootstrap as gb
from smishlens.evaluate import metrics_at


def test_percentile_ci_is_the_2_5_97_5_interval():
    ci = gb._percentile_ci([0.0, 1.0, 2.0, 3.0, 4.0])
    assert ci["lo"] == 0.1
    assert ci["hi"] == 3.9
    assert ci["n_valid"] == 5


def test_bootstrap_reports_point_metrics_and_cis(monkeypatch):
    monkeypatch.setattr(gb, "N_RESAMPLES", 40)
    y = np.asarray([1, 1, 0, 0, 1, 0, 0, 1], dtype=int)
    p = np.asarray([0.9, 0.6, 0.7, 0.2, 0.8, 0.3, 0.1, 0.4])
    tau = 0.5
    out = gb._bootstrap(y, p, tau, np.random.default_rng(0))

    point = metrics_at(y, p, tau)
    assert set(out) == {"recall", "fpr", "pr_auc"}
    for key, block in out.items():
        assert block["point"] == point[key]
        assert block["lo"] <= block["point"] <= block["hi"]
        assert block["n_valid"] <= 40


def test_bootstrap_skips_degenerate_resamples(monkeypatch):
    # A single-class bootstrap draw makes PR-AUC NaN; it must not enter the CI.
    monkeypatch.setattr(gb, "N_RESAMPLES", 5)
    y = np.asarray([1, 0], dtype=int)
    p = np.asarray([0.9, 0.1])
    out = gb._bootstrap(y, p, 0.5, np.random.default_rng(1))
    assert out["pr_auc"]["n_valid"] < 5
