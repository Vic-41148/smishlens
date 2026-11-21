"""Per-source adapters for the modern pool (Tier 2, Day 8).

Every adapter ends at ``smishlens.data.finalize``, so the unified schema
(``text, y, y3, source, lang, date, has_url, scam_type, synthetic``) is
guaranteed. Raw paths live under ``data/raw`` (git-ignored) and can be
redirected with ``SMISHLENS_RAW`` for tests.

Each source has one DATA_CARD.md under ``data/cards/``; the parsing decisions
made here are documented there, not re-explained per function.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd

from smishlens.data import finalize, load_uci

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("SMISHLENS_RAW", ROOT / "data" / "raw"))


def _path(*parts: str, raw: Path | None = None) -> Path:
    return (raw or RAW).joinpath(*parts)


def load_uci_from_raw(raw: Path | None = None) -> pd.DataFrame:
    """The Tier 1 anchor, read through the same path the anchor builder uses."""
    return load_uci(_path("uci", "SMSSpamCollection", raw=raw))


def load_moz(raw: Path | None = None) -> pd.DataFrame:
    """MOZ-Smishing: ``id, source, text, label``; labels Legitimate/Smishing."""
    df = pd.read_csv(_path("moz", "test.csv", raw=raw), encoding_errors="replace")
    label = df["label"].astype(str).str.strip().str.lower()
    y3 = label.map({"legitimate": "ham", "smishing": "smish"})
    out = pd.DataFrame({"text": df["text"].astype(str), "y3": y3}).dropna(subset=["y3"])
    out["lang"] = "pt"
    return finalize(out, "moz")


def load_exais(raw: Path | None = None) -> pd.DataFrame:
    """ExAIS: 20 ragged USER*.csv files, no reliable header.

    The column count changes per row because message text contains commas. The
    label is a literal ``HAM``/``SPAM`` cell; everything after it is message
    text, re-joined with ``", "``. Do not index by column position.
    """
    files = sorted(_path("exais", raw=raw).glob("extracted/**/*.csv"))
    if not files:
        raise FileNotFoundError(f"no ExAIS CSVs under {_path('exais', raw=raw)}")
    texts: list[str] = []
    labels: list[str] = []
    for path in files:
        # Line-based parsing: the field count differs per row, and pandas would
        # take its width from the first line and silently drop wider rows.
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                cells = [c.strip().strip('"') for c in line.rstrip("\r\n").split(",")]
                idx = next(
                    (i for i, c in enumerate(cells) if c.upper() in {"HAM", "SPAM"}),
                    None,
                )
                if idx is None:
                    continue
                text = ", ".join(c for c in cells[idx + 1 :] if c)
                if not text:
                    continue
                texts.append(text)
                labels.append(cells[idx].lower())
    out = pd.DataFrame({"text": texts, "y3": labels})
    out["y3"] = out["y3"].map({"ham": "ham", "spam": "spam"})
    out["lang"] = "en"
    return finalize(out, "exais")


def load_mishra(raw: Path | None = None) -> pd.DataFrame:
    """Mishra & Soni: ``LABEL, TEXT, ...`` with inconsistent label casing."""
    df = pd.read_csv(
        _path("mishra_soni", "Dataset_5971.csv", raw=raw), encoding_errors="replace"
    )
    label = df["LABEL"].astype(str).str.strip().str.lower()
    y3 = label.map({"ham": "ham", "spam": "spam", "smishing": "smish"})
    out = pd.DataFrame({"text": df["TEXT"].astype(str), "y3": y3}).dropna(subset=["y3"])
    return finalize(out, "mishra")


def load_imc25(raw: Path | None = None) -> pd.DataFrame:
    """IMC'25: positives only. ``time`` is a phone-UI string and is not parsed."""
    df = pd.read_csv(
        _path("imc25", "dataset", "final_dataset_output.csv", raw=raw),
        encoding_errors="replace",
        usecols=["text", "language", "scam_type"],
    )
    df = df[df["text"].notna()].copy()
    out = pd.DataFrame({"text": df["text"].astype(str)})
    out["y3"] = "smish"
    out["lang"] = df["language"].astype(str).str.lower()
    out["scam_type"] = df["scam_type"]
    return finalize(out, "imc25")


def load_smishtank(raw: Path | None = None) -> pd.DataFrame:
    """Smishtank Dataset I: positives only.

    ``Fulltext`` embeds UI headers and OCR line breaks; ``MainText`` is the
    message body, so that is what we use.
    """
    df = pd.read_csv(
        _path("smishtank", "analysisdataset.csv", raw=raw),
        encoding_errors="replace",
        usecols=["MainText", "timeReceived", "Message Categories", "Sender"],
    )
    df = df[df["MainText"].notna()].copy()
    out = pd.DataFrame({"text": df["MainText"].astype(str)})
    out["y3"] = "smish"
    out["date"] = df["timeReceived"]
    out["scam_type"] = df["Message Categories"]
    return finalize(out, "smishtank")


def load_wspr(raw: Path | None = None) -> pd.DataFrame:
    """WSPR Lab: phishing only. The header has 6 names but every row has 7
    fields (the campaign label is unheaded), so parse positionally with the
    stdlib reader: field 3 is the quoted message."""
    path = _path("wspr", "phishing_messages.csv", raw=raw)
    texts: list[str] = []
    dates: list[str] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if len(row) < 4 or not row[3].strip():
                continue
            texts.append(row[3])
            dates.append(row[5] if len(row) > 5 else "")
    out = pd.DataFrame({"text": texts, "date": dates})
    out["y3"] = "smish"
    return finalize(out, "wspr")


def load_africa(raw: Path | None = None) -> pd.DataFrame:
    """Africa HF set: fully synthetic (``is_synthetic == 1`` for every row)."""
    files = sorted(_path("hf", "africa-smishing", "data", raw=raw).glob("*.parquet"))
    if not files:
        raise FileNotFoundError("no Africa parquet under data/raw/hf/africa-smishing")
    df = pd.concat(
        [pd.read_parquet(f, columns=["sms_text", "label", "language"]) for f in files],
        ignore_index=True,
    )
    df = df[df["sms_text"].notna()].copy()
    out = pd.DataFrame({"text": df["sms_text"].astype(str)})
    out["y3"] = df["label"].astype(int).map({0: "ham", 1: "smish"})
    out["lang"] = df["language"].astype(str).str.lower()
    return finalize(out, "africa", synthetic=True)


LOADERS = {
    "moz": load_moz,
    "exais": load_exais,
    "mishra": load_mishra,
    "imc25": load_imc25,
    "smishtank": load_smishtank,
    "wspr": load_wspr,
    "africa": load_africa,
}
