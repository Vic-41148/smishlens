#!/usr/bin/env python3
"""Freeze the modern dev pool into ``data/processed/modern/``.

Writes:
    train.csv, cal.csv, test.csv   processed text (git-ignored: unclear licences)
    split_manifest.csv             row_id, source, split, labels, text hash — no text
    loso_folds.json                leave-one-source-out fold definitions
    data_audit.json                counts, dedup log, splits, dataset hash

The split is drawn once at seed 42. Re-running this script on the same raw tree
reproduces the same manifest; do not change the seed afterwards.

Usage: python data/build_modern_pool.py [--raw DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smishlens.data import SCHEMA  # noqa: E402
from smishlens.pool import build  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "modern"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = build(raw=Path(args.raw))

    frame, split_of = result["frame"], result["split_of"]
    for name in ("train", "cal", "test"):
        part = frame[split_of == name]
        part[SCHEMA].to_csv(
            out / f"{name}.csv", index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
        )

    result["manifest"].to_csv(
        out / "split_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
    )
    (out / "loso_folds.json").write_text(json.dumps(result["loso"], indent=2) + "\n")
    (out / "data_audit.json").write_text(json.dumps(result["audit"], indent=2) + "\n")

    audit = result["audit"]
    print(f"modern pool -> {out}")
    print(f"  rows              {audit['pool_rows']:>7}")
    print(f"  prevalence        {audit['pool_prevalence']:>7.4f}")
    for name, part in audit["splits"].items():
        print(
            f"  {name:<8}          {part['rows']:>7} rows  "
            f"{part['positives']} pos / {part['ham']} ham"
        )
    print(f"  dataset hash      {audit['dataset_sha256_16']}")
    print(f"  folds             {len(result['loso']['folds'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
