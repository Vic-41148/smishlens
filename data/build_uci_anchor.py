#!/usr/bin/env python3
"""Materialise the frozen UCI anchor split into ``data/processed/uci/``.

This script deliberately contains no data logic of its own: it calls the same
``load_uci`` -> ``clean_dedup`` -> ``split_pool`` path that ``smishlens.train``
uses, then writes the result to disk once. The split is frozen at seed 42; do not
re-run with a different seed and do not edit the CSVs by hand.

Usage: python data/build_uci_anchor.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smishlens.data import SCHEMA, clean_dedup, dataset_hash, load_uci, split_pool  # noqa: E402

RAW = ROOT / "data" / "raw" / "uci" / "SMSSpamCollection"
OUT = ROOT / "data" / "processed" / "uci"
SEED = 42


def main() -> None:
    raw = load_uci(RAW)
    frame = clean_dedup(raw)
    train, cal, test = split_pool(frame, seed=SEED)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("cal", cal), ("test", test)):
        part[SCHEMA].to_csv(
            OUT / f"{name}.csv", index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
        )

    total = len(frame)
    rows = [
        ("raw rows", len(raw)),
        ("exact duplicates dropped", len(raw) - total),
        ("label conflicts dropped", 0),
        ("rows after dedup", total),
        ("train", len(train)),
        ("cal", len(cal)),
        ("test", len(test)),
    ]
    print(f"UCI anchor -> {OUT}")
    for label, value in rows:
        print(f"  {label:<26} {value:>6}")
    for name, part in (("train", train), ("cal", cal), ("test", test)):
        spam = int(part["y"].sum())
        print(
            f"  {name:<26} {len(part):>6} rows  "
            f"{spam} spam / {len(part) - spam} ham  "
            f"{int(part['has_url'].sum())} with URL"
        )
    print(f"  dataset hash (all rows)    {dataset_hash(frame)}")
    print(f"  seed                       {SEED}")


if __name__ == "__main__":
    main()
