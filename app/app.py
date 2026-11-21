"""Streamlit demo. A thin skin over ``predict_one``; it shows the same fields.

Run: ``streamlit run app/app.py``
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from smishlens.attacks import ATTACKS
from smishlens.predict import load_model, predict_one

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")

st.set_page_config(page_title="smishlens", page_icon="🛡️", layout="wide")

SAMPLES = {
    "Ham": "Hey, are we still meeting at 6? I'll bring the notes.",
    "Smish": "Your parcel is held. Pay the redelivery fee today: hxxp://example[.]test/fee",
    "Tricky": "Are you free tonight? Did you win the match?",
    "Disguised": "Yоur pаrcel is held. Pаy the redelivery fee tоday: hxxp://example[.]test/fee",
}


@st.cache_resource
def bundle():
    return load_model(ARTIFACT_DIR)


def flag_text(flags: dict) -> str:
    parts = []
    if flags.get("confusables"):
        parts.append(f"{flags['confusables']} look-alike letter(s)")
    if flags.get("format_chars"):
        parts.append(f"{flags['format_chars']} hidden character(s)")
    if flags.get("fullwidth"):
        parts.append(f"{flags['fullwidth']} full-width character(s)")
    if flags.get("empty"):
        parts.append("empty message")
    if flags.get("error"):
        parts.append(f"internal error: {flags['error']}")
    return ", ".join(parts)


def show_result(pred, tau: float) -> None:
    if pred.action == "junk":
        st.error(f"🔴 SMISHING / SPAM · P(spam) = {pred.p_spam:.0%} · threshold {tau:.2f}")
    elif pred.action == "quarantine":
        st.warning(f"🟡 NOT SURE: quarantined for review · P(spam) = {pred.p_spam:.0%}")
    else:
        st.success(f"🟢 LOOKS OK: delivered · P(spam) = {pred.p_spam:.0%} · threshold {tau:.2f}")

    st.caption("Calibrated estimate, not a guarantee.")
    st.write(
        f"**Action:** {pred.action} · **Conformal set:** {pred.conformal_set} · "
        f"**Latency:** {pred.latency_ms:.1f} ms · **Model:** {pred.model_version}"
    )
    if pred.trigger_words:
        st.write("**Why:** " + " · ".join(f"{w} ({s:+.2f})" for w, s in pred.trigger_words))
    if pred.normaliser_flags:
        st.info(f"Disguise signals: {flag_text(pred.normaliser_flags)}")
    if pred.explanation:
        st.caption(f"Explanation (LLM-phrased, does not affect the verdict): {pred.explanation}")


def main() -> None:
    st.title("🛡️ smishlens")
    st.caption("Calibrated, explainable SMS spam and smishing detection.")

    try:
        b = bundle()
    except Exception as exc:  # no artifact yet
        st.error(f"No model loaded from `{ARTIFACT_DIR}`: {exc}")
        st.info("Train one with `python -m smishlens.train --out artifacts`.")
        return

    with st.sidebar:
        tau = st.slider(
            "Spam threshold", 0.05, 0.99, float(b.meta["threshold"]["value"]), 0.01
        )
        st.caption("🔒 Runs locally. Messages are not stored.")
        st.caption(f"Model: {b.meta.get('model_name')} v{b.meta.get('model_version')}")

    if "msg" not in st.session_state:
        st.session_state.msg = ""

    cols = st.columns(len(SAMPLES))
    for col, (name, value) in zip(cols, SAMPLES.items(), strict=True):
        if col.button(f"Try: {name}"):
            st.session_state.msg = value

    text = st.text_area("SMS message", key="msg", height=140, max_chars=5000)
    st.caption(f"{len(text)}/5000 characters")

    if st.button("🔍 Check message", type="primary"):
        if not text.strip():
            st.warning("Please enter a message.")
        else:
            show_result(predict_one(text, b, threshold=tau), tau)

    tab_evasion, tab_perf = st.tabs(["Evasion demo", "Performance"])

    with tab_evasion:
        st.write("Disguise a message and see what the normaliser notices.")
        attack = st.selectbox("Perturbation", list(ATTACKS))
        seed = st.number_input("Seed", 0, 999, 0, 1)
        if st.button("Disguise and score"):
            import random

            rng = random.Random(int(seed))
            disguised = ATTACKS[attack](text or SAMPLES["Smish"], rng)
            st.code(disguised, language=None)
            show_result(predict_one(disguised, b, threshold=tau), tau)

    with tab_perf:
        results = Path("reports/results")
        files = sorted(results.glob("*.json")) if results.exists() else []
        if not files:
            st.info("No result files yet. Run the evaluation to populate `reports/results/`.")
        for path in files:
            st.subheader(path.name)
            st.json(json.loads(path.read_text()))


if __name__ == "__main__":
    main()
