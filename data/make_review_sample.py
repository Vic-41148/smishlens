#!/usr/bin/env python3
"""Draw the label-audit sample for hand-checking (100 rows per pooled source).

The reviewer fills ``reviewer_label`` (and optionally ``reviewer_note``); the
file is deliberately git-ignored because it contains text from sources whose
licences do not allow redistribution.

Sampling: 100 rows per source, stratified by label where the source has both
classes, seed 42. Rows are drawn from the frozen pool, so a corrected label
here is a correction to the pool itself.

Usage: python data/make_review_sample.py [--per-source 100] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from smishlens.pool import POOL_SOURCES, build, row_id  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "processed" / "review" / "review_sample.csv"
POOL_DIR = ROOT / "data" / "processed" / "modern"


def sample(frame: pd.DataFrame, per_source: int, seed: int) -> pd.DataFrame:
    parts = []
    for src in POOL_SOURCES:
        sub = frame[frame["source"] == src]
        # Stratify by label so a rare ham side is not lost to chance.
        per_class = per_source // max(sub["y"].nunique(), 1)
        drawn = pd.concat(
            [
                g.sample(min(per_class, len(g)), random_state=seed)
                for _, g in sub.groupby("y")
            ]
        )
        if len(drawn) < per_source:
            rest = sub.drop(index=drawn.index)
            extra = rest.sample(min(per_source - len(drawn), len(rest)), random_state=seed)
            drawn = pd.concat([drawn, extra])
        parts.append(drawn.reset_index(drop=True))
    out = pd.concat(parts, ignore_index=True)
    out["row_id"] = [row_id(s, t) for s, t in zip(out["source"], out["text"], strict=True)]
    out["review_id"] = [f"R{i:03d}" for i in range(1, len(out) + 1)]
    out["reviewer_label"] = ""
    out["reviewer_note"] = ""
    return out[
        [
            "review_id",
            "row_id",
            "source",
            "text",
            "y3",
            "y",
            "lang",
            "scam_type",
            "reviewer_label",
            "reviewer_note",
        ]
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    # Use the committed corrections so a new sample is drawn from the frozen pool.
    overrides_path = POOL_DIR / "label_overrides.json"
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else None
    result = build(label_overrides=overrides)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample(result["frame"], args.per_source, args.seed).to_csv(out, index=False)
    print(f"review sample -> {out}")
    print(f"  rows {len(pd.read_csv(out))} across {len(POOL_SOURCES)} sources")
    print("  fill reviewer_label with ham | spam | smish, then tell me the disagreements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
