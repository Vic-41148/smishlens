"""Shortcut probes, LOSO harness and the evasion suite."""

import random

import numpy as np
import pandas as pd
import pytest

from smishlens.attacks import ATTACKS, eval_attack, strip_urls, url_slices
from smishlens.loso import loso, macro
from smishlens.probes import format_features, shortcut_report


def test_format_features_columns():
    f = format_features(["Hello  world!", "🎉 win £5"])
    assert {"n_chars", "n_words", "curly_quotes", "n_emoji"} <= set(f.columns)
    assert f.loc[1, "n_emoji"] == 1


def test_probe_flags_length_leak():
    rows = []
    for _ in range(150):
        rows.append({"text": "x" * 10, "y": 0, "source": "a"})
        rows.append({"text": "x" * 200, "y": 1, "source": "a"})
    rep = shortcut_report(pd.DataFrame(rows), min_per_class=20)
    assert rep["length_only_auc"] > 0.95


def test_probe_clean_case():
    rng = np.random.default_rng(0)
    rows = [
        {"text": "x" * int(rng.integers(20, 200)), "y": int(rng.integers(0, 2)), "source": "a"}
        for _ in range(400)
    ]
    rep = shortcut_report(pd.DataFrame(rows), min_per_class=20)
    assert rep["length_only_auc"] < 0.62


def test_source_identifiability_from_curly_quotes():
    rows = []
    for i in range(100):
        rows.append({"text": f"plain message {i}", "y": i % 2, "source": "a"})
        rows.append({"text": f"\u2018curly\u2019 message {i}", "y": i % 2, "source": "b"})
    rep = shortcut_report(pd.DataFrame(rows))
    assert rep["source_id_acc_from_format"] > rep["source_id_chance"] + 0.2


def _loso_frame():
    rows = []
    for src, n_pos, n_neg in [("a", 20, 20), ("b", 20, 20), ("c", 20, 0), ("d", 4, 0)]:
        for i in range(n_pos):
            rows.append({"text": f"{src} spam {i}", "y": 1, "source": src})
        for i in range(n_neg):
            rows.append({"text": f"{src} ham {i}", "y": 0, "source": src})
    return pd.DataFrame(rows)


def test_loso_modes():
    def fp(train, test):
        return np.full(len(test), 0.9), 0.5

    res = loso(_loso_frame(), fp, min_pos=10, min_neg=10)
    modes = dict(zip(res["source"], res["mode"], strict=True))
    assert modes == {"a": "both", "b": "both", "c": "recall_only", "d": "too_small"}
    assert macro(res)["folds"] == 2


def test_loso_train_sources_restriction():
    def fp(train, test):
        assert set(train["source"]) <= {"a"}
        return np.full(len(test), 0.9), 0.5

    res = loso(_loso_frame(), fp, train_sources=["a"], min_pos=10, min_neg=10)
    assert len(res) >= 1


@pytest.mark.parametrize("name", list(ATTACKS))
def test_attacks_change_text(name):
    out = ATTACKS[name]("pay the fee now", random.Random(0))
    assert isinstance(out, str)


def test_eval_attack_bounds():
    spam = ["you won a prize", "claim your free gift"]

    def fn(texts):
        p = np.array([1.0 if "prize" in t else 0.0 for t in texts])
        return np.c_[1 - p, p]

    mean, std = eval_attack(fn, spam, ATTACKS["homoglyph"], 0.5)
    assert 0 <= mean <= 1
    assert std >= 0


def test_url_slices():
    df = pd.DataFrame({"text": ["a", "b", "c"], "y": [1, 1, 0], "has_url": [True, False, False]})
    slices = url_slices(df)
    assert len(slices["spam_with_url"]) == 1
    assert len(slices["spam_linkless"]) == 1
    assert len(slices["ham"]) == 1
    assert "http" not in strip_urls("go http://x.test now")
