"""Leakage-proof classical pipeline.

Custom transformers MUST live in an importable module or joblib cannot unpickle
them elsewhere.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from smishlens.normalise import PHONE, SHORT, URL, normalise


class CleanText(BaseEstimator, TransformerMixin):
    """Normalise (and optionally strip URLs for the E10 masked arm)."""

    def __init__(self, enabled: bool = True, mask: bool = True, strip_urls: bool = False):
        # sklearn contract: store constructor params as-is, no validation here
        self.enabled = enabled
        self.mask = mask
        self.strip_urls = strip_urls

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        out = []
        for item in X:
            text = str(item)
            if self.strip_urls:
                text = URL.sub(" ", text)
            out.append(normalise(text, mask=self.mask) if self.enabled else text)
        return out

    def get_feature_names_out(self, input_features=None):  # noqa: N803
        return np.asarray(["text"], dtype=object)


class MetaFeatures(BaseEstimator, TransformerMixin):
    """Formatting and shape features, computed on RAW text."""

    NAMES = [
        "n_chars",
        "n_words",
        "digit_ratio",
        "caps_ratio",
        "n_excl",
        "n_currency",
        "has_url",
        "has_shortcode",
        "has_phone",
        "nonascii_ratio",
        "longest_word",
        "n_caps_words",
    ]

    def __init__(self, use_url: bool = True):
        self.use_url = use_url

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        rows = []
        for item in X:
            text = str(item)
            n = max(len(text), 1)
            words = text.split()
            n_words = max(len(words), 1)
            alpha = max(sum(c.isalpha() for c in text), 1)
            rows.append(
                [
                    len(text),
                    len(words),
                    sum(c.isdigit() for c in text) / n,
                    sum(c.isupper() for c in text) / alpha,
                    text.count("!"),
                    sum(text.count(c) for c in "£$€₹"),
                    float(self.use_url and bool(URL.search(text))),
                    float(bool(SHORT.search(text))),
                    float(bool(PHONE.search(text))),
                    sum(not c.isascii() for c in text) / n,
                    max((len(w) for w in words), default=0),
                    sum(1 for w in words if w.isupper() and len(w) > 2) / n_words,
                ]
            )
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):  # noqa: N803
        return np.asarray(self.NAMES, dtype=object)


def build_features(
    normalise_on: bool = True,
    strip_urls: bool = False,
    min_df_word: int = 2,
    min_df_char: int = 3,
) -> FeatureUnion:
    """Word TF-IDF + char TF-IDF + formatting features, fitted inside the pipeline."""
    kwargs = {"enabled": normalise_on, "strip_urls": strip_urls}
    word = Pipeline(
        [
            ("clean", CleanText(**kwargs)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=min_df_word,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
        ]
    )
    char = Pipeline(
        [
            ("clean", CleanText(**kwargs)),
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=min_df_char, sublinear_tf=True)),
        ]
    )
    meta = Pipeline([("m", MetaFeatures(use_url=not strip_urls)), ("s", MaxAbsScaler())])
    return FeatureUnion([("word", word), ("char", char), ("meta", meta)])


def build_final(C: float = 0.5, **feat_kw) -> Pipeline:
    """Calibration is the LAST pipeline step: one object to save, load, test and explain."""
    return Pipeline(
        [
            ("feats", build_features(**feat_kw)),
            (
                "clf",
                CalibratedClassifierCV(
                    LinearSVC(C=C, class_weight="balanced", dual="auto"),
                    method="sigmoid",
                    cv=5,
                ),
            ),
        ]
    )
