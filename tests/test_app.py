"""Streamlit smoke test with a stub bundle; no real message or pool row is used."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from smishlens.serve import save_bundle

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


class _Feats:
    def transform(self, texts):
        return np.ones((len(texts), 1), dtype=float)


class _Clf:
    def decision_function(self, X):
        return np.full(len(X), 10.0, dtype=float)


def _stub(tmp_path: Path) -> Path:
    bundle = {
        "feats": _Feats(),
        "clf": _Clf(),
        "platt": {"a": 1.0, "b": 0.0},
        "q_hat": {"0": 0.2, "1": 0.9},
        "meta": {"test_or_gold_used": False},
    }
    save_bundle(
        tmp_path,
        bundle,
        {"test_or_gold_used": False, "alpha": 0.05, "q_hat": {"0": 0.2, "1": 0.9}},
    )
    return tmp_path


def test_app_renders_notice_and_scores_a_message(tmp_path, monkeypatch):
    _stub(tmp_path)
    monkeypatch.setenv("SMISHLENS_SERVE_ARTIFACT", str(tmp_path))
    monkeypatch.delenv("SMISHLENS_ENABLE_MMBERT", raising=False)

    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()

    assert not at.exception
    assert any("Triage aid, not a verdict" in w.value for w in at.warning)
    assert at.title[0].value == "🛡️ smishlens"

    at.text_area[0].set_value("Your parcel is held, pay the fee now.")
    at.button[0].click().run()

    assert not at.exception
    assert any("likely scam" in e.value for e in at.error)
    assert any("Conformal set" in m.label for m in at.metric)


def test_demo_button_fills_the_message_box(tmp_path, monkeypatch):
    _stub(tmp_path)
    monkeypatch.setenv("SMISHLENS_SERVE_ARTIFACT", str(tmp_path))
    monkeypatch.delenv("SMISHLENS_ENABLE_MMBERT", raising=False)

    from smishlens.demo_examples import EXAMPLES

    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception

    first = EXAMPLES[0]
    demo = next(b for b in at.button if b.label == first["label"])
    demo.click().run()

    assert not at.exception
    assert at.text_area[0].value == first["text"]
