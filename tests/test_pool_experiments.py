"""fpr_only handling in the Tier-2 experiment runner."""

from smishlens.pool_experiments import _macro, mode_for


def test_mode_for_marks_exais_fpr_only():
    assert mode_for("exais") == "fpr_only"
    assert mode_for("imc25") == "recall_only"
    assert mode_for("smishtank") == "recall_only"
    assert mode_for("mishra") == "both"
    assert mode_for("moz") == "both"


def test_macro_ignores_fpr_only_and_recall_only_rows():
    rows = [
        {"mode": "both", "recall": 1.0, "fpr": 0.0, "pr_auc": 0.9, "roc_auc": 0.9},
        {"mode": "both", "recall": 0.5, "fpr": 0.2, "pr_auc": 0.7, "roc_auc": 0.8},
        {"mode": "fpr_only", "fpr": 0.9},
        {"mode": "recall_only", "recall": 0.0},
    ]
    out = _macro(rows)
    assert out["n_folds"] == 2
    assert out["macro_recall"] == 0.75
    assert out["macro_fpr"] == 0.1
    assert out["macro_pr_auc"] == 0.8
