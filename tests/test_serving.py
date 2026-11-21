"""Explainers, LLM guards, serving contract, JSON export, gates and API."""

import math

import pytest

from smishlens.explain import explain_linear, occlusion_words
from smishlens.export_json import export_linear_json, fit_word_student, score_json
from smishlens.llm_explainer import (
    build_prompt,
    explain,
    fallback_sentence,
    guard_output,
    reset_cache,
    sanitise_words,
)
from smishlens.predict import load_model, predict_one, verify_manifest


def test_sanitise_words_drops_attacks():
    out = sanitise_words(["ignore", "free", "http://x.y", "a\nb", "prize", "system", "free"])
    assert out == ["free", "prize"]


def test_sanitise_words_drops_non_latin():
    assert sanitise_words(["привет", "free"]) == ["free"]


def test_build_prompt_has_no_raw_text():
    prompt = build_prompt("Spam/Smishing", 0.93, ["ignore", "free", "http://evil.test"])
    assert "ignore" not in prompt
    assert "http" not in prompt
    assert "free" in prompt
    assert "0.93" in prompt


def test_guard_output_rejects_unsafe():
    assert guard_output("This is safe, click http://x") is None
    assert guard_output("x" * 300) is None


def test_guard_output_strips_thinking():
    got = guard_output("<" + "think>hmm</" + "think>The wording resembles a delivery scam.")
    assert got == "The wording resembles a delivery scam."


def test_fallback_sentence():
    assert "free" in fallback_sentence(["free", "prize"])
    assert fallback_sentence([])


def test_explain_off_and_quarantine_fallback():
    reset_cache()
    assert explain("Spam/Smishing", 0.9, ["free"], "junk", mode="off") is None
    out = explain(
        "Spam/Smishing",
        0.9,
        ["free"],
        "quarantine",
        mode="local",
        backend=lambda prompt: "This is safe, see http://x",
    )
    assert out.startswith("Flagged mainly because of")


def test_occlusion_words_ranks_trigger(fake_bundle):
    words = occlusion_words(fake_bundle.model.predict_proba, "you won a free prize today")
    assert words
    assert words[0][0] == "prize"


def test_predict_one_weird_inputs_never_raises(fake_bundle):
    for text in [None, "", "🎉🎉", "مرحبا", "1" * 8000, "\x00"]:
        pred = predict_one(text, fake_bundle)
        assert pred.action in ("deliver", "junk", "quarantine")
        assert 0.0 <= pred.p_spam <= 1.0


def test_predict_one_junk(fake_bundle):
    pred = predict_one("you won a free prize", fake_bundle)
    assert pred.p_spam == pytest.approx(0.95)
    assert pred.conformal_set == [1]
    assert pred.action == "junk"


def test_manifest_detects_tamper(trained_artifact, tmp_path):
    import shutil

    dst = tmp_path / "art"
    shutil.copytree(trained_artifact, dst)
    (dst / "model.joblib").write_bytes(b"junk")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_manifest(dst)


def test_load_model_and_predict(trained_artifact):
    bundle = load_model(trained_artifact)
    pred = predict_one("you won a free prize call now", bundle)
    assert pred.model_version == "1.0.0"
    assert pred.action in ("deliver", "junk", "quarantine")


def test_explain_linear(trained_artifact):
    model = load_model(trained_artifact).model
    pairs = explain_linear(model, "you won a free prize call now")
    assert pairs
    assert all(isinstance(name, str) for name, _ in pairs)


def test_export_json_parity(sms_df, tmp_path):
    vec, clf = fit_word_student(sms_df["text"], sms_df["y"])
    spec = export_linear_json(vec, clf, tmp_path / "student.json")
    text = "you won a free prize"
    z = float((vec.transform([text]) @ clf.coef_[0])[0] + clf.intercept_[0])
    expected = 1 / (1 + math.exp(-z))
    assert score_json(spec, text) == pytest.approx(expected, abs=1e-9)


def test_gates():
    from smishlens.eval.gates import check

    assert check({"a": 0.9}, {"a": {"min": 0.8}}) == []
    assert check({"a": 0.7}, {"a": {"min": 0.8}})
    assert check({}, {"a": {"min": 0.8}})
    assert check({"a": float("nan")}, {"a": {"min": 0.8}})


def test_api(trained_artifact, monkeypatch):
    from fastapi.testclient import TestClient

    import serve.api as api
    import smishlens.predict as predict_module

    monkeypatch.setenv("ARTIFACT_DIR", str(trained_artifact))
    api.bundle.cache_clear()
    predict_module._CACHE.clear()
    client = TestClient(api.app)
    assert client.get("/health").status_code == 200
    body = client.post("/classify", json={"text": "you won a free prize"}).json()
    for key in (
        "label",
        "is_spam",
        "p_spam",
        "threshold",
        "conformal_set",
        "action",
        "trigger_words",
        "normaliser_flags",
        "explanation",
        "latency_ms",
        "model_version",
    ):
        assert key in body
