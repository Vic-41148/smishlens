"""E-recal threshold helper on synthetic scores (no message text)."""

from __future__ import annotations

import numpy as np
import pytest

import smishlens.e_recal as er


def test_recal_uses_k_ham_to_set_the_quantile_threshold():
    # 100 ham with distinct scores and 100 spam well above them: the 99th
    # percentile of 25 sampled ham sits just below its max, so only that max
    # row is a calibration FP and no remaining ham crosses the threshold.
    y = np.asarray([1] * 100 + [0] * 100, dtype=int)
    p = np.asarray([2.0] * 100 + list(np.linspace(0.0, 1.0, 100)), dtype=float)
    out = er._recal(p, y, k=25, rng=np.random.default_rng(0))

    assert out["k"] == 25
    assert out["n_eval"] == 175
    assert out["calibration_fpr"]["mean"] == pytest.approx(0.04)
    assert out["recall"]["mean"] == pytest.approx(1.0)
    # Some ham outside the sample can sit above the sampled max, so the eval FPR
    # is small but not necessarily zero.
    assert 0.0 <= out["fpr"]["mean"] < 0.1


def test_recal_reports_min_and_max_over_resplits():
    y = np.asarray([1] * 50 + [0] * 50, dtype=int)
    p = np.linspace(0.0, 1.0, 100)
    out = er._recal(p, y, k=10, rng=np.random.default_rng(3))
    for key in ("calibration_fpr", "recall", "fpr"):
        assert out[key]["min"] <= out[key]["mean"] + 1e-12
        assert out[key]["mean"] <= out[key]["max"] + 1e-12


def test_frozen_row_raises_on_unknown_model(tmp_path):
    path = tmp_path / "e4.json"
    path.write_text('{"rows": [{"model": "svm", "source": "moz"}]}')
    with pytest.raises(KeyError):
        er._frozen_row(path, "mmbert_small", "moz")
