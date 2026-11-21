"""Streamlit capstone: pool svm triage with a frozen conformal rule.

Run: ``streamlit run app/app.py``

The hosted path is svm only. mmBERT-base is optional and local-only, behind
``SMISHLENS_ENABLE_MMBERT=1``. No message is stored or logged.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from smishlens.demo_examples import EXAMPLES
from smishlens.serve import (
    BAND_HAM,
    BAND_REVIEW,
    BAND_SCAM,
    DEFAULT_DIR,
    MISSING_BUNDLE_MESSAGE,
    BundleVersionError,
    load_bundle,
    predict_message,
)

NOTICE = (
    "Triage aid, not a verdict. In our synthetic gold check about half of "
    "legitimate bank/service messages were flagged; the pool has almost no real "
    "bank-alert ham; and the model is weak on non-English held-out sources. "
    "Nothing you type is stored or logged."
)

BAND_ICON = {BAND_SCAM: "🔴", BAND_REVIEW: "🟡", BAND_HAM: "🟢"}


def _band_box(result) -> None:
    icon = BAND_ICON.get(result.band, "⚪")
    if result.band == BAND_SCAM:
        st.error(f"{icon} **{result.band}**")
    elif result.band == BAND_HAM:
        st.success(f"{icon} **{result.band}**")
    else:
        st.warning(f"{icon} **{result.band}**")
    st.caption(
        "Band comes only from the frozen class-conditional conformal set. "
        "It is not a hard verdict."
    )


def _result_panel(result) -> None:
    left, mid, right = st.columns(3)
    left.metric("Calibrated P(spam)", f"{result.p_spam:.1%}")
    mid.metric("P(ham)", f"{result.p_ham:.1%}")
    names = {0: "ham", 1: "spam"}
    shown = "{" + ", ".join(names[c] for c in result.conformal_set) + "}"
    right.metric("Conformal set", shown)

    st.write(
        f"**Message stats:** {result.n_chars:,} chars · {result.n_words:,} words · "
        f"{result.n_urls:,} URL(s) · digit ratio {result.digit_ratio:.2f}"
    )
    if result.truncated:
        st.caption("Input was truncated to the first 5,000 characters.")
    if result.urls:
        st.write("**Extracted URL(s):**")
        for url in result.urls:
            st.code(url, language=None)
    else:
        st.caption("No URL found in the message.")


def _load_mmbert():
    """Local-only second opinion. Never required by the hosted app."""
    import json

    from smishlens.train_mmbert import MMBertScorer

    root = Path(__file__).resolve().parents[1]
    art = root / "artifacts" / "mmbert" / "e3_base"
    meta = json.loads((art / "metadata.json").read_text())
    return MMBertScorer.load(art / "mmbert", T=float(meta["calibration"]["T"]))


@st.cache_resource
def bundle_for(path: str) -> dict:
    return load_bundle(path)


def _use_demo(text: str) -> None:
    """Callback so the text-area state is set before the widget is created."""
    st.session_state["message"] = text


@st.cache_resource
def mmbert_for(path: str):
    return _load_mmbert()


def main() -> None:
    st.set_page_config(page_title="smishlens", page_icon="🛡️", layout="wide")
    st.title("🛡️ smishlens")
    st.caption("Pool-trained svm triage for SMS spam and smishing.")

    st.warning(NOTICE, icon="⚠️")

    path = os.getenv("SMISHLENS_SERVE_ARTIFACT", str(DEFAULT_DIR))
    try:
        bundle = bundle_for(path)
    except FileNotFoundError:
        st.error(MISSING_BUNDLE_MESSAGE)
        st.info(
            "The public repository ships code and docs only. To build the bundle "
            "you need your own licensed copy of the pool data; then run "
            "`make build-serve` and point `SMISHLENS_SERVE_ARTIFACT` at it."
        )
        return
    except BundleVersionError as exc:
        st.error(str(exc))
        st.info(
            "Install the pinned `requirements.txt` versions, then rebuild with "
            "`make build-serve`."
        )
        return
    except Exception as exc:  # tampered artifact or other load failure
        st.error(f"No verified pool artifact at `{path}`: {exc}")
        st.info("Build one with `make build-serve`.")
        return

    text = st.text_area(
        "Message",
        key="message",
        height=160,
        placeholder="Paste one SMS message. It is scored locally and not stored.",
    )
    sender = st.text_input(
        "Sender ID (experimental)",
        key="sender",
        help="Shown for context only. It does not change the score; there is no evidence for it yet.",
    )
    if sender:
        st.caption("Sender ID is displayed only. The model does not read it (experimental).")

    if st.button("Check message", key="check", type="primary"):
        if not text.strip():
            st.warning("Enter a message first.")
        else:
            result = predict_message(bundle, text)
            _band_box(result)
            _result_panel(result)
            st.caption(f"Scored in {result.latency_ms:.1f} ms on this machine.")

            if os.getenv("SMISHLENS_ENABLE_MMBERT") == "1":
                try:
                    scorer = mmbert_for("e3_base")
                    p_mm = float(scorer.predict_proba([text])[0, 1])
                    st.info(
                        f"Local mmBERT-base second opinion: P(spam) = {p_mm:.1%}. "
                        "This arm is optional and is not part of the hosted app."
                    )
                except Exception as exc:
                    st.caption(f"Optional mmBERT arm unavailable: {exc}")

    with st.expander("Synthetic demo messages (invented for this app)"):
        st.caption("These are not gold rows and not pool rows. They are labelled synthetic.")
        for i, row in enumerate(EXAMPLES):
            st.button(
                row["label"],
                key=f"demo_{i}",
                on_click=_use_demo,
                args=(row["text"],),
            )

    st.divider()
    meta = bundle.get("meta", {})
    st.caption(
        "Frozen recipe: pool train → LinearSVC; Platt and conformal `q_hat` on "
        "`cal_a` only. No test or gold row was used. "
        f"Artifact `{meta.get('artifact_sha256', 'unknown')[:16]}` · "
        f"α={meta.get('alpha')} · q_hat={meta.get('q_hat')}"
    )


if __name__ == "__main__":
    main()
