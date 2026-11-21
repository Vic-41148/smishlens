#!/usr/bin/env python3
"""Freeze the modern dev pool into ``data/processed/modern/``.

Writes:
    train.csv, cal.csv, test.csv   processed text (git-ignored: unclear licences)
    split_manifest.csv             row_id, source, split, labels, text hash — no text
    loso_folds.json                leave-one-source-out fold definitions
    data_audit.json                counts, dedup log, splits, dataset hash

The split is drawn once at seed 42. Re-running this script on the same raw tree
reproduces the same manifest; do not change the seed afterwards.

Usage: python data/build_modern_pool.py [--raw DIR] [--out DIR] [--overrides FILE]

``--overrides`` is a JSON object of ``{row_id: ham|spam|smish}`` produced from a
reviewed label audit (see ``data/apply_audit.py``). Corrections are applied
before dedup, so the pool is rebuilt from the corrected labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smishlens.pool import build, write_pool  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "modern"))
    ap.add_argument("--overrides", default=None, help="JSON {row_id: y3} corrections")
    ap.add_argument(
        "--dedup-exclusions",
        default=None,
        help="JSON {row_id: reason} audited near-dups missed by the rule",
    )
    args = ap.parse_args()

    overrides = None
    out = Path(args.out)
    override_path = Path(args.overrides) if args.overrides else out / "label_overrides.json"
    if override_path.exists():
        overrides = json.loads(override_path.read_text())
        print(f"  overrides         {override_path}")

    dedup_exclusions = None
    exclusion_path = (
        Path(args.dedup_exclusions) if args.dedup_exclusions else out / "dedup_exclusions.json"
    )
    if exclusion_path.exists():
        dedup_exclusions = json.loads(exclusion_path.read_text())
        print(f"  dedup exclusions  {exclusion_path}")

    result = build(
        raw=Path(args.raw),
        label_overrides=overrides,
        dedup_exclusions=dedup_exclusions,
    )
    write_pool(result, out)

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
    if overrides:
        print(f"  label overrides   {audit['label_overrides']['n']}")
    if dedup_exclusions:
        print(f"  dedup exclusions  {audit['dedup_exclusions']['applied']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
