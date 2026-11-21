"""Label-audit helpers: parse a reviewed sample and measure disagreement.

The review file is drawn by ``data/make_review_sample.py``. A blank
``reviewer_label`` means the reviewer agrees with the stored label; anything
else is a correction. Disagreement is binary (ham vs malicious); a spam-vs-smish
mismatch is counted separately because it changes ``y3`` only.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

LABELS = {
    "ham": "ham",
    "legit": "ham",
    "legitimate": "ham",
    "spam": "spam",
    "smish": "smish",
    "smishing": "smish",
    "phishing": "smish",
}

REQUIRED = {"review_id", "row_id", "source", "text", "y", "y3", "reviewer_label"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def parse_label(value) -> str | None:
    """Return a canonical y3, or None for 'agree' (blank / NaN)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in ("", "nan", "none"):
        return None
    if text not in LABELS:
        raise ValueError(f"unrecognised reviewer_label {value!r} (use ham, spam or smish)")
    return LABELS[text]


def load_review(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["reviewer_label"] = df["reviewer_label"].map(parse_label)
    df["y3"] = df["y3"].astype(str).str.strip().str.lower()
    df["y"] = df["y"].astype(int)
    df["reviewer_y3"] = df["reviewer_label"]
    df["reviewer_y"] = df["reviewer_label"].map(
        lambda v: None if v is None else int(v != "ham")
    )
    return df


def summarise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-source disagreement rates with Wilson intervals, plus the rows.

    Every row of the review file is a reviewed row: a blank label means the
    reviewer agreed. Only the explicitly corrected rows are disagreeing, so the
    denominator is the whole per-source sample, not just the labelled rows.
    """
    df = df.copy()
    labelled = df["reviewer_label"].notna()
    df["disagrees_binary"] = labelled & (df["reviewer_y"] != df["y"])
    df["disagrees_subtype"] = (
        labelled
        & (df["reviewer_y"] == df["y"])
        & (df["reviewer_y"] == 1)
        & (df["reviewer_y3"] != df["y3"])
    )
    rows = []
    for src in sorted(df["source"].unique()):
        part = df[df["source"] == src]
        n = len(part)
        k = int(part["disagrees_binary"].sum())
        lo, hi = wilson(k, n)
        rows.append(
            {
                "source": src,
                "reviewed": n,
                "corrected": int(part["reviewer_label"].notna().sum()),
                "disagree": k,
                "rate": k / n if n else float("nan"),
                "wilson_lo": lo,
                "wilson_hi": hi,
                "subtype_only": int(part["disagrees_subtype"].sum()),
            }
        )
    disagree = df[df["disagrees_binary"] | df["disagrees_subtype"]]
    return pd.DataFrame(rows), disagree
