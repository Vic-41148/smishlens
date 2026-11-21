"""Leakage-proof classical pipeline.

Custom transformers MUST live in an importable module or joblib cannot unpickle
them elsewhere.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from smishlens.normalise import PHONE, SHORT, URL, deleet_mixed, normalise

STOPWORDS = frozenset(ENGLISH_STOP_WORDS)


@lru_cache(maxsize=1)
def _stemmer():
    """English Snowball stemmer, imported lazily so the default pipeline needs no extra package."""
    import snowballstemmer

    return snowballstemmer.stemmer("english")


class CleanText(BaseEstimator, TransformerMixin):
    """Normalise, then optionally deleet, remove stopwords and stem.

    Every switch is off by default except normalisation, so the default pipeline
    is unchanged. ``confusables`` selects the normaliser's mixed/global fold.
    """

    def __init__(
        self,
        enabled: bool = True,
        mask: bool = True,
        strip_urls: bool = False,
        confusables: str = "mixed",
        deleet: bool = False,
        stopwords: bool = False,
        stem: bool = False,
    ):
        # sklearn contract: store constructor params as-is, no validation here
        self.enabled = enabled
        self.mask = mask
        self.strip_urls = strip_urls
        self.confusables = confusables
        self.deleet = deleet
        self.stopwords = stopwords
        self.stem = stem

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        stemmer = _stemmer() if self.stem else None
        out = []
        for item in X:
            text = str(item)
            if self.strip_urls:
                text = URL.sub(" ", text)
            if self.enabled:
                text = normalise(text, mask=self.mask, mode=self.confusables)
            if self.deleet:
                text = deleet_mixed(text)
            if stemmer is not None:
                # leave masked placeholders (<url>, <num>) untouched
                text = " ".join(
                    w if not w.isalpha() else stemmer.stemWord(w) for w in text.split()
                )
            if self.stopwords:
                text = " ".join(w for w in text.split() if w not in STOPWORDS)
            out.append(text)
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


def word_tfidf(min_df: int = 2, **clean_kw) -> Pipeline:
    """Word 1-2 gram TF-IDF behind the shared cleaner."""
    return Pipeline(
        [
            ("clean", CleanText(**clean_kw)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=min_df,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
        ]
    )


def char_tfidf(min_df: int = 3, **clean_kw) -> Pipeline:
    """Char 2-5 gram TF-IDF behind the shared cleaner."""
    return Pipeline(
        [
            ("clean", CleanText(**clean_kw)),
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=min_df,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def bow(min_df: int = 2, **clean_kw) -> Pipeline:
    """Plain word 1-2 gram counts (E2 ablation arm)."""
    return Pipeline(
        [
            ("clean", CleanText(**clean_kw)),
            (
                "counts",
                CountVectorizer(ngram_range=(1, 2), min_df=min_df, token_pattern=r"(?u)\b\w+\b"),
            ),
        ]
    )


def meta_block(use_url: bool = True) -> Pipeline:
    """Formatting and shape features, computed on raw text."""
    return Pipeline([("m", MetaFeatures(use_url=use_url)), ("s", MaxAbsScaler())])


BLOCKS = {
    "bow": lambda clean_kw, min_df_word, min_df_char: bow(min_df_word, **clean_kw),
    "word": lambda clean_kw, min_df_word, min_df_char: word_tfidf(min_df_word, **clean_kw),
    "char": lambda clean_kw, min_df_word, min_df_char: char_tfidf(min_df_char, **clean_kw),
    "meta": lambda clean_kw, min_df_word, min_df_char: meta_block(
        use_url=not clean_kw.get("strip_urls", False)
    ),
}


def build_features(
    normalise_on: bool = True,
    strip_urls: bool = False,
    min_df_word: int = 2,
    min_df_char: int = 3,
    blocks: tuple[str, ...] = ("word", "char", "meta"),
    confusables: str = "mixed",
    deleet: bool = False,
    stopwords: bool = False,
    stem: bool = False,
) -> FeatureUnion:
    """Word TF-IDF + char TF-IDF + formatting features, fitted inside the pipeline.

    ``blocks`` selects the arms for the E2 feature ablation; the default keeps
    the production pipeline identical to v1.0.
    """
    unknown = set(blocks) - set(BLOCKS)
    if unknown:
        raise ValueError(f"unknown feature block(s): {sorted(unknown)}")
    clean_kw = {
        "enabled": normalise_on,
        "strip_urls": strip_urls,
        "confusables": confusables,
        "deleet": deleet,
        "stopwords": stopwords,
        "stem": stem,
    }
    return FeatureUnion(
        [
            (name, BLOCKS[name](clean_kw, min_df_word, min_df_char))
            for name in blocks
        ]
    )


def build_final(C: float = 0.5, **feat_kw) -> Pipeline:
    """Calibration is the LAST pipeline step: one object to save, load, test and explain."""
    return Pipeline(
        [
            ("feats", build_features(**feat_kw)),
            (
                "clf",
                CalibratedClassifierCV(
                    # random_state pins liblinear's dual coordinate-descent
                    # permutation; without it tau drifts ~1e-6 per retrain.
                    LinearSVC(C=C, class_weight="balanced", dual="auto", random_state=42),
                    method="sigmoid",
                    cv=5,
                ),
            ),
        ]
    )
