"""The cached fold features must never see the held-out source."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from smishlens.pool_experiments import (
    build_fold_matrices,
    evaluate_cached,
    oof_cached,
    pipe_for,
    transform_with,
)


def _frame(texts, labels, source):
    return pd.DataFrame({"text": texts, "y": labels, "source": source})


def test_vectoriser_vocabulary_comes_only_from_train_rows():
    train = _frame(
        [f"hello there friend number {i}" if i % 2 else f"see you later tonight {i}" for i in range(10)],
        [i % 2 for i in range(10)],
        ["moz"] * 10,
    )
    held = _frame(["qzjxqzjxqzjx qzjxqzjxqzjx"], [1], ["mishra"])
    mats = build_fold_matrices(train, held)

    vocab = set()
    for _, block in mats.feats.transformer_list:
        if hasattr(block, "named_steps") and "tfidf" in block.named_steps:
            vocab |= set(block.named_steps["tfidf"].vocabulary_)

    # No n-gram of a token that appears only in the held-out source is known.
    assert not [t for t in vocab if "qzjx" in t or "zjxq" in t or "jxqz" in t]
    # Train vocabulary is there, so the check is not vacuous.
    assert [t for t in vocab if "friend" in t]
    assert transform_with(mats, train["text"]).nnz > 0


def test_cached_final_fit_matches_the_plain_pipeline():
    texts = [f"claim your prize now code {i} http://x.io/{i}" for i in range(40)]
    texts += [f"hello mum see you at six love {i}" for i in range(40)]
    labels = [1] * 40 + [0] * 40
    frame = _frame(texts, labels, ["moz"] * 80)
    test = _frame(texts[:8], labels[:8], ["mishra"] * 8)
    mats = build_fold_matrices(frame, test)

    cached = evaluate_cached(mats, "svm")["p"]
    direct = pipe_for("svm").fit(texts, np.array(labels)).decision_function(test["text"])
    # The final fit vectorises all train rows either way, so it must be exact.
    assert np.array_equal(cached, direct)


def test_cached_oof_keeps_the_ranking():
    texts = [f"claim your prize now code {i} http://x.io/{i}" for i in range(40)]
    texts += [f"hello mum see you at six love {i}" for i in range(40)]
    labels = np.array([1] * 40 + [0] * 40)
    mats = build_fold_matrices(_frame(texts, labels, ["moz"] * 80))

    cached = oof_cached(mats, "svm")
    direct = np.empty(80)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(80), labels):
        pipe = pipe_for("svm").fit([texts[i] for i in tr], labels[tr])
        direct[va] = pipe.decision_function([texts[i] for i in va])

    # One vectoriser per fold instead of one per CV split: scores may move a
    # little, the ordering may not.
    assert float(np.corrcoef(cached, direct)[0, 1]) > 0.99
