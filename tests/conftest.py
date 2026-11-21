"""Shared fixtures: a synthetic SMS frame and one trained artifact."""

import numpy as np
import pandas as pd
import pytest

COLUMNS = ["text", "y", "y3", "source", "lang", "date", "has_url", "scam_type", "synthetic"]

HAM = [
    "hey are we still meeting at {n} tonight",
    "can you send the notes from class {n}",
    "dinner at {n} works for me",
    "thanks for the update see you at {n}",
]
SPAM = [
    "you won a free prize call now to claim cash {n}",
    "urgent your parcel is held pay the fee {n}",
    "congratulations claim your free gift card now {n}",
    "verify your account immediately {n}",
]


def make_frame() -> pd.DataFrame:
    rows = []
    for i in range(120):
        rows.append({"text": HAM[i % len(HAM)].format(n=i), "y": 0, "y3": "ham"})
    for i in range(120):
        rows.append({"text": SPAM[i % len(SPAM)].format(n=i), "y": 1, "y3": "spam"})
    df = pd.DataFrame(rows)
    df["source"] = "synth"
    df["lang"] = "en"
    df["date"] = np.nan
    df["has_url"] = False
    df["scam_type"] = np.nan
    df["synthetic"] = False
    return df.sample(frac=1, random_state=0).reset_index(drop=True)[COLUMNS]


@pytest.fixture(scope="session")
def sms_df():
    return make_frame()


@pytest.fixture(scope="session")
def trained_artifact(tmp_path_factory, sms_df):
    from smishlens.data import clean_dedup, split_pool
    from smishlens.train import train_classical

    out = tmp_path_factory.mktemp("artifact")
    train, cal, _ = split_pool(clean_dedup(sms_df))
    train_classical(train, cal, out, C=1.0)
    return out


class FakeModel:
    """Scores 0.95 when the text contains 'prize', else 0.05."""

    def predict_proba(self, texts):
        p = np.array([0.95 if "prize" in str(t).lower() else 0.05 for t in texts], dtype=float)
        return np.c_[1 - p, p]


@pytest.fixture
def fake_bundle():
    from smishlens.predict import Bundle

    meta = {
        "model_version": "test-1",
        "threshold": {"value": 0.5},
        "conformal": {"q_hat": {"0": 0.2, "1": 0.2}},
    }
    return Bundle(FakeModel(), meta)
