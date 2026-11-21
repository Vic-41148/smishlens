"""Pool serve module: bands from the frozen rule, URL/stats, artifact round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from smishlens.serve import (
    BAND_HAM,
    BAND_REVIEW,
    BAND_SCAM,
    MISSING_BUNDLE_MESSAGE,
    BundleVersionError,
    band_for_set,
    extract_urls,
    load_bundle,
    predict_message,
    save_bundle,
)


class _Feats:
    def transform(self, texts):
        return np.ones((len(texts), 1), dtype=float)


class _Clf:
    def __init__(self, score: float):
        self.score = score

    def decision_function(self, X):
        return np.full(len(X), self.score, dtype=float)


def _bundle(score: float, q0: float = 0.2, q1: float = 0.9) -> dict:
    return {
        "feats": _Feats(),
        "clf": _Clf(score),
        "platt": {"a": 1.0, "b": 0.0},
        "q_hat": {"0": q0, "1": q1},
        "meta": {"test_or_gold_used": False},
    }


def test_band_for_set_is_exactly_the_frozen_rule():
    assert band_for_set([1]) == BAND_SCAM
    assert band_for_set([0]) == BAND_HAM
    assert band_for_set([]) == BAND_REVIEW
    assert band_for_set([0, 1]) == BAND_REVIEW


def test_extract_urls_dedupes_in_order():
    text = "go to hxxp://a.example/x and hxxp://a.example/x or www.b.example"
    assert extract_urls(text) == ["hxxp://a.example/x", "www.b.example"]


def test_likely_scam_from_high_probability():
    out = predict_message(_bundle(10.0), "your parcel is held, pay now")
    assert out.band == BAND_SCAM
    assert out.conformal_set == [1]
    assert 0.99 < out.p_spam <= 1.0


def test_looks_ordinary_from_low_probability():
    out = predict_message(_bundle(-10.0, q0=0.9, q1=0.2), "see you at six")
    assert out.band == BAND_HAM
    assert out.conformal_set == [0]


def test_needs_review_from_empty_set():
    out = predict_message(_bundle(0.0, q0=0.2, q1=0.2), "unclear message")
    assert out.band == BAND_REVIEW
    assert out.conformal_set == []


def test_empty_input_does_not_raise():
    out = predict_message(_bundle(0.0), "")
    assert out.n_chars == 0
    assert out.n_words == 0
    assert out.urls == []
    assert out.band in {BAND_SCAM, BAND_HAM, BAND_REVIEW}


def test_very_long_input_is_truncated():
    out = predict_message(_bundle(0.0), "x" * 6000)
    assert out.truncated is True
    assert out.n_chars == 5000


def test_unicode_and_hinglish_do_not_raise():
    for text in ("आपका खाता बंद हो गया है", "Aapka account block ho gaya hai, jaldi karein"):
        out = predict_message(_bundle(0.0), text)
        assert out.n_chars == len(text)
        assert 0.0 <= out.p_spam <= 1.0


def test_save_and_load_roundtrip(tmp_path):
    metadata = {
        "test_or_gold_used": False,
        "alpha": 0.05,
        "q_hat": {"0": 0.2, "1": 0.9},
    }
    save_bundle(tmp_path, _bundle(10.0), metadata)
    loaded = load_bundle(tmp_path)
    out = predict_message(loaded, "pay now")
    assert out.band == BAND_SCAM
    assert loaded["meta"]["sklearn_version"]
    assert loaded["meta"]["joblib_version"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "MANIFEST.sha256",
        "metadata.json",
        "svm_bundle.joblib",
    ]


def test_load_refuses_a_bundle_that_used_test_or_gold(tmp_path):
    save_bundle(tmp_path, _bundle(10.0), {"test_or_gold_used": True})
    with pytest.raises(RuntimeError):
        load_bundle(tmp_path)


def test_missing_bundle_has_a_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="not distributed in this repo"):
        load_bundle(tmp_path)
    assert "make build-serve" in MISSING_BUNDLE_MESSAGE


def test_load_refuses_a_different_sklearn_version(tmp_path):
    save_bundle(
        tmp_path,
        _bundle(10.0),
        {"test_or_gold_used": False, "sklearn_version": "0.0.0"},
    )
    with pytest.raises(BundleVersionError, match="rebuild"):
        load_bundle(tmp_path)


def test_load_refuses_a_different_joblib_version(tmp_path):
    save_bundle(
        tmp_path,
        _bundle(10.0),
        {"test_or_gold_used": False, "joblib_version": "0.0.0"},
    )
    with pytest.raises(BundleVersionError, match="rebuild"):
        load_bundle(tmp_path)
