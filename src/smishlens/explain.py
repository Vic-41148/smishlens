"""Token attribution: linear coefficients for the SVM, occlusion for any model.

The words shown to the user are raw words. Any words sent to an LLM are
sanitised first (see ``llm_explainer``).
"""

from __future__ import annotations

import numpy as np


def explain_linear(model, raw_text: str, k: int = 6) -> list[tuple[str, float]]:
    """Top-k (feature, contribution) pairs for the calibrated linear pipeline.

    Coefficients are averaged over the CV folds inside ``CalibratedClassifierCV``
    so one unlucky fold cannot dominate the explanation.
    """
    feats = model.named_steps["feats"]
    clf = model.named_steps["clf"]
    coef = np.mean([cc.estimator.coef_[0] for cc in clf.calibrated_classifiers_], axis=0)
    names = feats.get_feature_names_out()
    row = feats.transform([raw_text]).tocsr()
    row.sum_duplicates()
    contributions = row.data * coef[row.indices]
    pairs = [(str(names[j]), float(v)) for j, v in zip(row.indices, contributions, strict=True)]
    return sorted(pairs, key=lambda item: -abs(item[1]))[:k]


def occlusion_words(proba_fn, text: str, k: int = 6, max_words: int = 60) -> list[tuple[str, float]]:
    """Drop each word, re-score, rank by how much ``p_spam`` falls.

    One batched call, so this works for any scorer including an encoder. Only
    words whose removal lowers the spam probability are reported.
    """
    words = str(text).split()[:max_words]
    if not words:
        return []
    variants = [" ".join(words)] + [
        " ".join(words[:i] + words[i + 1 :]) for i in range(len(words))
    ]
    p = np.asarray(proba_fn(variants), dtype=float)[:, 1]
    drops = p[0] - p[1:]
    ranked = np.argsort(-drops)[:k]
    return [(words[i], float(drops[i])) for i in ranked if drops[i] > 0]
