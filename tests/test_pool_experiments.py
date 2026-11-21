"""fpr_only handling and mmBERT wiring in the Tier-2 experiment runner."""

import numpy as np
import pandas as pd

import smishlens.pool_experiments as pe
from smishlens.pool_experiments import _macro, mode_for
from smishlens.train import mmbert_label


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


def test_mmbert_label_is_derived_from_the_checkpoint_name():
    assert mmbert_label("models/mmBERT-base") == "mmbert_base_temp_scaled"
    assert mmbert_label("models/mmBERT-small") == "mmbert_small_temp_scaled"


class _StubScorer:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)

    def predict_proba(self, texts):
        assert len(texts) == len(self.p)
        return np.c_[1 - self.p, self.p]


def _e3_frame():
    rows = [
        ("a", 1, "mishra", "train"),
        ("b", 0, "moz", "train"),
        ("c", 1, "mishra", "cal"),
        ("d", 0, "moz", "cal"),
        ("e", 1, "mishra", "test"),
        ("f", 0, "mishra", "test"),
        ("g", 1, "moz", "test"),
        ("h", 0, "moz", "test"),
        ("i", 0, "exais", "test"),
        ("j", 0, "exais", "test"),
        ("k", 1, "imc25", "test"),
        ("l", 1, "imc25", "test"),
    ]
    return pd.DataFrame(rows, columns=["text", "y", "source", "split"])


def test_run_e3_mmbert_builds_per_source_and_macro_rows(monkeypatch, tmp_path):
    calls = {}

    def fake_fit(train, cal, out_dir, model_key, epochs=3, max_len=128, bs=16):
        calls["train"] = train.copy()
        calls["cal"] = cal.copy()
        calls["out_dir"] = out_dir
        return _StubScorer([0.9, 0.1, 0.8, 0.2, 0.3, 0.4, 0.7, 0.6]), 0.5, 0.4, {
            "calibration": {"T": 1.0},
            "torch": {},
        }

    monkeypatch.setattr(pe, "_fit_mmbert", fake_fit)
    payload = pe.run_e3_mmbert(_e3_frame(), "small", tmp_path)

    assert payload["model"] == "mmbert_small"
    assert set(calls["train"]["split"]) == {"train"}
    assert set(calls["cal"]["split"]) == {"cal"}
    by_source = {r["source"]: r for r in payload["rows"] if r["source"] != "macro"}
    assert by_source["exais"]["mode"] == "fpr_only"
    assert by_source["exais"]["fpr"] == 0.0
    assert by_source["imc25"]["mode"] == "recall_only"
    assert by_source["imc25"]["recall"] == 1.0
    assert by_source["mishra"]["mode"] == "both"
    macro = next(r for r in payload["rows"] if r["source"] == "macro")
    assert macro["n_folds"] == 2
    assert macro["macro_recall"] == 1.0
    assert macro["macro_fpr"] == 0.0
    assert macro["macro_pr_auc"] == 1.0


def _e4_frame():
    rows = []
    for source in ("mishra", "moz", "exais", "imc25", "smishtank"):
        rows.append((f"{source}-train", 1 if source in ("mishra", "imc25", "smishtank") else 0, source, "train"))
        rows.append((f"{source}-cal", 1 if source in ("mishra", "imc25", "smishtank") else 0, source, "cal"))
    for source in ("mishra", "moz"):
        rows.extend([(f"{source}-pos", 1, source, "test"), (f"{source}-neg", 0, source, "test")])
    rows.extend([("exais-neg1", 0, "exais", "test"), ("exais-neg2", 0, "exais", "test")])
    rows.extend([("imc25-pos1", 1, "imc25", "test"), ("imc25-pos2", 1, "imc25", "test")])
    rows.extend([("smishtank-pos1", 1, "smishtank", "test"), ("smishtank-pos2", 1, "smishtank", "test")])
    return pd.DataFrame(rows, columns=["text", "y", "source", "split"])


def test_run_e4_mmbert_holds_out_the_source_from_train_and_cal(monkeypatch, tmp_path):
    calls = {}

    class _LenScorer:
        def predict_proba(self, texts):
            return np.c_[np.full(len(texts), 0.6), np.full(len(texts), 0.4)]

    def fake_fit(train, cal, out_dir, model_key, epochs=3, max_len=128, bs=16):
        held = str(out_dir).rsplit("heldout_", 1)[-1]
        calls[held] = {"train": train.copy(), "cal": cal.copy()}
        return _LenScorer(), 0.5, 0.4, {}

    monkeypatch.setattr(pe, "_fit_mmbert", fake_fit)
    monkeypatch.setattr(pe, "HELD_OUT", ("mishra", "moz", "exais"))
    payload = pe.run_e4_mmbert(_e4_frame(), "small", tmp_path)

    assert set(calls) == {"mishra", "moz", "exais"}
    for held, frames in calls.items():
        assert held not in set(frames["train"]["source"])
        assert held not in set(frames["cal"]["source"])
    macro = next(r for r in payload["rows"] if r["source"] == "macro")
    assert macro["n_folds"] == 2
