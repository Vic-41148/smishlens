"""E7/E8/E10 helpers: identity attack, E3 aggregation and the raw mmBERT path."""

from __future__ import annotations

import json
import types
from functools import partial

import numpy as np
import pandas as pd

from smishlens.e7_e10 import (
    _attack_recall,
    _both_class_macro,
    _e3_reference,
    _raw_mmbert_scores,
    url_obfuscation,
)


def test_url_obfuscation_p_zero_is_the_identity():
    texts = [
        "hello there",
        "visit http://example.test/a",
        "go to https://bank.example/login now",
    ]
    rng = __import__("random").Random(0)
    assert [url_obfuscation(t, rng, p=0) for t in texts] == texts


def test_identity_attack_reproduces_clean_recall():
    texts = ["one", "two two", "three three three", "four four four four"]
    score_fn = lambda ts: np.asarray([len(t) for t in ts], dtype=float)  # noqa: E731
    tau = 8.0
    clean = float(np.mean(score_fn(texts) >= tau))
    result = _attack_recall(score_fn, texts, partial(url_obfuscation, p=0), tau)
    assert result["recall_mean"] == clean
    assert result["recall_std"] == 0.0
    assert result["n"] == len(texts)


def test_both_class_macro_uses_only_mishra_and_moz():
    frame = pd.DataFrame(
        {
            "source": ["mishra", "mishra", "moz", "moz", "exais"],
            "y": [1, 0, 1, 0, 0],
        }
    )
    p = np.asarray([0.9, 0.1, 0.8, 0.6, 0.99])
    recall, fpr = _both_class_macro(frame, p, tau=0.5)
    assert recall == (1.0 + 1.0) / 2
    assert fpr == (0.0 + 1.0) / 2


def test_e3_reference_reads_macro_and_tau(tmp_path):
    path = tmp_path / "e3.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {"model": "svm", "source": "mishra", "tau": 0.25},
                    {"model": "svm", "source": "macro", "macro_recall": 0.8, "macro_fpr": 0.02},
                ]
            }
        )
    )
    ref = _e3_reference(path, "svm")
    assert ref == {"tau": 0.25, "recall": 0.8, "fpr": 0.02}


class _FakeEnc(dict):
    def to(self, dev):
        assert dev == "cpu"
        return self


class _FakeTok:
    def __init__(self, max_len):
        self.max_len = max_len
        self.seen = []

    def __call__(self, texts, **kw):
        assert kw["truncation"] is True
        assert kw["max_length"] == self.max_len
        self.seen.append(list(texts))
        import torch

        return _FakeEnc(input_ids=torch.zeros((len(texts), 1)))


class _FakeModel:
    def eval(self):
        return self

    def __call__(self, **enc):
        import torch

        n = enc["input_ids"].shape[0]
        return types.SimpleNamespace(logits=torch.tensor([[0.0, 2.0]]).repeat(n, 1))


def test_raw_mmbert_scores_uses_the_scorer_contract():
    scorer = types.SimpleNamespace(
        dev="cpu", tok=_FakeTok(max_len=8), model=_FakeModel(), max_len=8, T=2.0
    )
    out = _raw_mmbert_scores(scorer, ["a", "b"])
    expected = 1 / (1 + np.exp(-2.0 / 2.0))
    assert out.shape == (2,)
    assert np.allclose(out, expected)
    assert scorer.tok.seen == [["a", "b"]]
    assert _raw_mmbert_scores(scorer, []).size == 0
