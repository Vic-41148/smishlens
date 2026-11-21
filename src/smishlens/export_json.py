"""Word-only linear student -> JSON, so a browser can score it in ~20 lines of JS.

Parity target: ``score_json`` must match ``TfidfVectorizer.transform`` plus the
SVM decision function to floating-point tolerance. Fit on the same raw text the
browser sees; the browser does no URL masking, which is a documented limitation.
"""

from __future__ import annotations

import json
import math
import re


def fit_word_student(texts, y, C: float = 1.0, ngram_range=(1, 2), min_df: int = 2):
    """Fit the word-only TF-IDF + LinearSVC student that gets exported."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    X = vec.fit_transform([str(t) for t in texts])
    clf = LinearSVC(C=C, class_weight="balanced", dual="auto").fit(X, y)
    return vec, clf


def export_linear_json(vec, clf, path):
    """Write the vocabulary, IDF weights and coefficients as one JSON file."""
    spec = {
        "vocab": {k: int(v) for k, v in vec.vocabulary_.items()},
        "idf": vec.idf_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "ngram": list(vec.ngram_range),
        "token_pattern": vec.token_pattern,
        "sublinear_tf": bool(vec.sublinear_tf),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh)
    return spec


def score_json(spec, text: str) -> float:
    """Reproduce the linear student's probability without scikit-learn."""
    tokens = re.findall(spec["token_pattern"], str(text).lower())
    lo, hi = spec["ngram"]
    terms = []
    for n in range(lo, hi + 1):
        terms += [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    tf: dict[str, int] = {}
    for term in terms:
        if term in spec["vocab"]:
            tf[term] = tf.get(term, 0) + 1

    weights = {}
    for term, count in tf.items():
        idf = spec["idf"][spec["vocab"][term]]
        weights[term] = ((1 + math.log(count)) if spec["sublinear_tf"] else count) * idf

    norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
    z = spec["intercept"] + sum(
        spec["coef"][spec["vocab"][term]] * value / norm for term, value in weights.items()
    )
    return 1 / (1 + math.exp(-z))
