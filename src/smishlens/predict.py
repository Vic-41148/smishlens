"""ONE inference entry point shared by Streamlit, FastAPI, the bot and the tests.

``predict_one`` never raises: any internal failure returns ``deliver`` with an
error flag, because a crashed filter must not block OTPs. Normalisation happens
inside the model object, so training and serving cannot drift.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib

from smishlens import llm_explainer
from smishlens.conformal import conformal_set
from smishlens.explain import occlusion_words
from smishlens.normalise import normalise_with_flags
from smishlens.policy import decide

MAX_CHARS = 5000


@dataclass(frozen=True)
class Bundle:
    """One model plus the metadata that governs it."""

    model: object  # anything with predict_proba(list[str]) -> (n, 2)
    meta: dict


@dataclass(frozen=True)
class Prediction:
    """The contract every surface returns."""

    label: str
    is_spam: bool
    p_spam: float
    threshold: float
    conformal_set: list
    action: str
    trigger_words: list
    normaliser_flags: dict
    explanation: str | None
    latency_ms: float
    model_version: str


def make_manifest(art_dir) -> Path:
    """SHA-256 every artifact so a tampered pickle is refused at load time."""
    d = Path(art_dir)
    lines = []
    for f in sorted(d.rglob("*")):
        if f.is_file() and f.name != "MANIFEST.sha256":
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            lines.append(f"{digest}  {f.relative_to(d).as_posix()}")
    path = d / "MANIFEST.sha256"
    path.write_text("\n".join(lines) + "\n")
    return path


def verify_manifest(art_dir) -> None:
    """Every listed file must hash-match.

    A joblib/pickle load is code execution, so never load an artifact you did
    not train.
    """
    d = Path(art_dir)
    manifest = d / "MANIFEST.sha256"
    if not manifest.exists():
        raise RuntimeError(f"no manifest in {d}")
    entries = [
        line.split(None, 1) for line in manifest.read_text().splitlines() if line.strip()
    ]
    if not entries:
        raise RuntimeError("empty manifest")
    for digest, rel in entries:
        target = d / rel.strip()
        if not target.exists():
            raise RuntimeError(f"artifact missing: {rel.strip()}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"artifact hash mismatch: {rel.strip()}, refusing to load")


_CACHE: dict[str, Bundle] = {}


def load_model(art_dir="artifacts") -> Bundle:
    """Verify the manifest, then load once and cache."""
    key = str(art_dir)
    if key in _CACHE:
        return _CACHE[key]
    d = Path(art_dir)
    verify_manifest(d)
    meta = json.loads((d / "metadata.json").read_text())
    if (d / "model.joblib").exists():
        model = joblib.load(d / "model.joblib")
    else:
        from smishlens.train_mmbert import MMBertScorer

        model = MMBertScorer.load(d / "mmbert", T=meta.get("calibration", {}).get("T", 1.0))
    _CACHE[key] = Bundle(model, meta)
    return _CACHE[key]


def _result(label, p, tau, S, action, words, flags, expl, t0, ver) -> Prediction:
    return Prediction(
        label=label,
        is_spam=label == "Spam/Smishing",
        p_spam=float(p),
        threshold=float(tau),
        conformal_set=list(S),
        action=action,
        trigger_words=words,
        normaliser_flags=flags,
        explanation=expl,
        latency_ms=(time.perf_counter() - t0) * 1000,
        model_version=ver,
    )


def predict_one(
    text,
    bundle: Bundle,
    threshold: float | None = None,
    explainer: str = "off",
    with_words: bool = True,
) -> Prediction:
    """Classify one message. Never raises."""
    t0 = time.perf_counter()
    ver = str(bundle.meta.get("model_version", "?"))
    try:
        raw = ("" if text is None else str(text))[:MAX_CHARS]
        tau = float(threshold if threshold is not None else bundle.meta["threshold"]["value"])
        if not raw.strip():
            return _result("Not Spam", 0.0, tau, [], "deliver", [], {"empty": 1}, None, t0, ver)

        _, flags = normalise_with_flags(raw)
        p = float(bundle.model.predict_proba([raw])[0, 1])

        q_hat = bundle.meta.get("conformal", {}).get("q_hat")
        if q_hat:
            S = conformal_set(p, {int(k): float(v) for k, v in q_hat.items()})
        else:  # no conformal info: fall back to the threshold alone
            S = [1] if p >= tau else [0]
        action = decide(p, tau, S)

        words = occlusion_words(bundle.model.predict_proba, raw) if with_words else []
        label = "Spam/Smishing" if p >= tau else "Not Spam"
        expl = llm_explainer.explain(label, p, [w for w, _ in words], action, mode=explainer)
        return _result(label, p, tau, S, action, words, flags, expl, t0, ver)
    except Exception as exc:  # fail OPEN: a filter crash must not block OTPs
        return _result(
            "Not Spam", 0.0, 1.0, [], "deliver", [], {"error": type(exc).__name__}, None, t0, ver
        )
