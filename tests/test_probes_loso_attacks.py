"""Shortcut probes, LOSO harness and the evasion suite."""

import random

import numpy as np
import pandas as pd
import pytest

from smishlens.attacks import (
    ATTACKS,
    augment_with_attacks,
    eval_attack,
    eval_families,
    family_summary,
    strip_urls,
    url_arm,
    url_slices,
)
from smishlens.loso import loso, macro, write_results
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


def test_write_results_json_per_source(tmp_path):
    import json

    def fp(train, test):
        return np.full(len(test), 0.9), 0.5

    res = loso(_loso_frame(), fp, min_pos=10, min_neg=10)
    path = write_results(res, name="loso_test", out_dir=tmp_path, meta={"model": "fake"})
    payload = json.loads(path.read_text())

    assert payload["meta"] == {"model": "fake"}
    by_source = {row["source"]: row for row in payload["per_source"]}
    assert by_source["a"]["mode"] == "both"
    assert by_source["c"]["mode"] == "recall_only"
    assert "fpr" not in by_source["c"]
    assert by_source["d"]["mode"] == "too_small"
    assert payload["macro"]["folds"] == 2
    assert payload["macro"]["recall"] == 1.0


def test_write_results_null_for_missing_metric(tmp_path):
    import json

    res = pd.DataFrame(
        [{"source": "tiny", "mode": "too_small", "n_pos": 1, "n_neg": 0, "recall": float("nan")}]
    )
    path = write_results(res, name="tiny", out_dir=tmp_path)
    text = path.read_text()
    assert "NaN" not in text and "null" not in text
    payload = json.loads(text)
    assert "recall" not in payload["per_source"][0]
    assert payload["macro"]["folds"] == 0


def test_augment_with_attacks_only_touches_positives():
    train = pd.DataFrame(
        {
            "text": ["win a prize", "claim cash", "see you soon", "lunch at one"],
            "y": [1, 1, 0, 0],
            "source": "a",
        }
    )
    out = augment_with_attacks(train, families=("leet",), copies=1, seed=0)
    assert len(out) == 6
    assert out["y"].sum() == 4
    ham = out[out["y"] == 0]
    assert ham["text"].tolist() == ["see you soon", "lunch at one"]
    spam = out[out["y"] == 1]["text"].tolist()
    assert "win a prize" in spam  # original rows are kept
    assert any("0" in t or "@" in t or "1" in t or "3" in t or "$" in t for t in spam)


def test_augment_with_attacks_rejects_unknown_family():
    train = pd.DataFrame({"text": ["win"], "y": [1], "source": ["a"]})
    with pytest.raises(ValueError):
        augment_with_attacks(train, families=("nope",))


def test_eval_families_groups_in_and_cross():
    spam = ["you won a prize", "claim your free gift"]

    def fn(texts):
        p = np.array([1.0 if "prize" in t else 0.0 for t in texts])
        return np.c_[1 - p, p]

    table = eval_families(fn, spam, 0.5, trained_on=("leet",))
    assert len(table) == len(ATTACKS)
    assert set(table["group"]) == {"in_family", "cross_family"}
    assert table.loc[table["attack"] == "leet", "group"].iloc[0] == "in_family"
    assert table.loc[table["attack"] == "homoglyph", "group"].iloc[0] == "cross_family"
    summary = family_summary(table)
    assert set(summary["group"]) == {"in_family", "cross_family"}
    assert summary["attacks"].sum() == len(ATTACKS)


def test_url_arm_shows_link_dependence():
    df = pd.DataFrame(
        {
            "text": ["claim http://x.test now", "claim your prize now", "see you at six"],
            "y": [1, 1, 0],
            "has_url": [True, False, False],
        }
    )

    def fn(texts):
        p = np.array([1.0 if ("prize" in t or "http" in t) else 0.0 for t in texts])
        return np.c_[1 - p, p]

    table = url_arm(fn, df, 0.5).set_index("view")
    assert table.loc["normal", "recall"] == 1.0
    assert table.loc["normal", "fpr"] == 0.0
    assert table.loc["url_masked", "recall"] == pytest.approx(0.5)
    assert table.loc["spam_linkless", "n_pos"] == 1
