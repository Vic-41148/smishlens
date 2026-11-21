"""Classical pipeline and evaluation metrics."""

import numpy as np
import pytest

from smishlens.evaluate import (
    boot_ci,
    mcnemar_exact,
    metrics_at,
    ppv,
    ppv_table,
    threshold_for_fpr,
)
from smishlens.features import build_features, build_final
from smishlens.models import tune


def test_ppv_worked_example():
    assert ppv(0.9866, 0.0082, 0.5) == pytest.approx(0.9918, abs=1e-3)


def test_ppv_table():
    rows = ppv_table(0.98, 0.01)
    assert len(rows) == 4
    assert rows[0]["prevalence"] == 0.5
    assert 0 <= rows[-1]["ppv"] <= 1


def test_threshold_for_fpr_respects_budget():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    _tau, tpr, fpr = threshold_for_fpr(y, p, 0.25)
    assert fpr <= 0.25
    assert 0 <= tpr <= 1


def test_metrics_at_perfect():
    m = metrics_at([0, 1], [0.1, 0.9], 0.5)
    assert m["recall"] == 1.0
    assert m["fpr"] == 0.0
    assert m["f1"] == 1.0


def test_boot_ci_perfect():
    lo, hi = boot_ci([0, 1] * 20, [0.1, 0.9] * 20, 0.5)
    assert lo == hi == 1.0


def test_mcnemar_exact():
    y = np.array([1] * 6)
    b, c, p = mcnemar_exact(y, y, [0] * 6)
    assert (b, c) == (6, 0)
    assert p < 0.05


def test_build_features_shape(sms_df):
    X = build_features().fit_transform(sms_df["text"], sms_df["y"])
    assert X.shape[0] == len(sms_df)
    assert X.shape[1] > 0


def test_build_final_predicts_probabilities(sms_df):
    model = build_final(C=1.0).fit(sms_df["text"], sms_df["y"])
    proba = model.predict_proba(sms_df["text"][:5])
    assert proba.shape == (5, 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_tune_returns_best_params(sms_df):
    search = tune("cnb", sms_df["text"], sms_df["y"], min_df_word=1, min_df_char=2)
    assert "clf__alpha" in search.best_params_
