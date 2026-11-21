#!/usr/bin/env python3
"""Verify the local raw-data tree and print the counts quoted in data/README.md.

Requires: pandas, pyarrow (for the Hugging Face parquet files).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent / "raw"


def uci() -> None:
    lines = (RAW / "uci" / "SMSSpamCollection").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    pairs = [line.split("\t", 1) for line in lines if "\t" in line]
    labels = Counter(label for label, _ in pairs)
    texts = [text for _, text in pairs]
    print(
        f"UCI            {len(pairs):>6} rows; {labels['ham']:,} ham / "
        f"{labels['spam']:,} spam; {len(pairs) - len(set(texts))} exact duplicates "
        f"({len(set(texts)):,} unique)"
    )


def imc() -> None:
    df = pd.read_csv(RAW / "imc25" / "dataset" / "final_dataset_output.csv")
    parsed = pd.to_datetime(df["time"], errors="coerce").notna().sum()
    print(
        f"IMC'25         {len(df):>6} rows; text non-null {df['text'].notna().sum():,}; "
        f"English {(df['language'] == 'English').sum():,}; "
        f"{parsed} absolute dates (relative/OCR timestamps -> no time split)"
    )


def mishra() -> None:
    df = pd.read_csv(
        RAW / "mishra_soni" / "Dataset_5971.csv", encoding_errors="replace"
    )
    lab = df["LABEL"].str.lower()
    print(
        f"Mishra & Soni  {len(df):>6} rows; {Counter(lab)['ham']:,} ham / "
        f"{Counter(lab)['spam']:,} spam / {Counter(lab)['smishing']:,} smishing"
    )


def moz() -> None:
    df = pd.read_csv(RAW / "moz" / "test.csv")
    print(f"MOZ            {len(df):>6} rows; {dict(Counter(df['label']))}")


def exais() -> None:
    spam = ham = 0
    for path in (RAW / "exais" / "extracted").rglob("*.csv"):
        text = path.read_text(encoding="utf-8", errors="replace")
        spam += sum(1 for line in text.splitlines() if ",SPAM," in line)
        ham += sum(1 for line in text.splitlines() if ",HAM," in line)
    print(f"ExAIS          {spam + ham:>6} rows; {ham:,} ham / {spam:,} spam")


def smishtank() -> None:
    data = json.loads((RAW / "smishtank" / "smishtank_smishes.json").read_text())
    print(f"Smishtank      {len(data.get('smishes') or []):>6} records via public API")


def wspr() -> None:
    msgs = pd.read_csv(
        RAW / "wspr" / "phishing_messages.csv", encoding_errors="replace"
    )
    camps = pd.read_csv(
        RAW / "wspr" / "phishing_campaigns.csv", encoding_errors="replace"
    )
    print(f"WSPR           {len(msgs):>6} messages / {len(camps):,} campaigns")


def hf() -> None:
    india = pd.read_parquet(RAW / "hf" / "india-spam-sms" / "data")
    print(f"India HF       {len(india):>6} rows; {dict(Counter(india['label']))}")

    africa = pd.read_parquet(RAW / "hf" / "africa-smishing" / "data")
    print(
        f"Africa HF      {len(africa):>6} rows; {dict(Counter(africa['label']))}; "
        f"synthetic={dict(Counter(africa['is_synthetic']))}"
    )

    combined = pd.read_parquet(
        RAW / "hf" / "sms-spam-combined" / "data" / "train-00000-of-00001.parquet"
    )
    print(
        f"Combined HF    {len(combined):>6} train rows; {dict(Counter(combined['label']))}"
    )


def main() -> None:
    uci()
    imc()
    mishra()
    moz()
    exais()
    smishtank()
    wspr()
    hf()


if __name__ == "__main__":
    main()
