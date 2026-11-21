#!/usr/bin/env python3
"""Fold the reviewed label-audit sample back into the modern pool.

Reads ``data/processed/review/review_sample.csv`` (drawn by
``data/make_review_sample.py``). A blank ``reviewer_label`` means the reviewer
agrees with the stored label. Any other value is a correction.

The report is per source, with a Wilson 95% interval. If any source disagrees on
more than 5% of its reviewed rows the script **stops without writing anything**
and prints the rows, because a source that bad is a decision, not a fix.

``--apply`` rebuilds the pool from the corrected labels (corrections are applied
before dedup), re-freezes the splits and LOSO folds at the same seed, writes the
new dataset hash into ``data_audit.json`` and appends a verification-log entry
that supersedes the old hash.

Usage:
    python data/apply_audit.py                 # report only
    python data/apply_audit.py --apply         # rebuild if the gate passes
    python data/apply_audit.py --review FILE --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from smishlens.audit import load_review, summarise  # noqa: E402
from smishlens.pool import DROPPED_SOURCES, POOL_SOURCES, build, write_pool  # noqa: E402

DEFAULT_REVIEW = ROOT / "data" / "processed" / "review" / "review_sample.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "modern"
LOG = ROOT / "reports" / "verification_log.md"
GATE = 0.05


def print_report(summary: pd.DataFrame) -> None:
    print(f"{'source':<12}{'reviewed':>9}{'disagree':>10}{'rate':>8}  95% Wilson")
    for row in summary.itertuples():
        print(
            f"{row.source:<12}{row.reviewed:>9}{row.disagree:>10}"
            f"{row.rate:>8.1%}  [{row.wilson_lo:.1%}, {row.wilson_hi:.1%}]"
            + (f"  (+{row.subtype_only} spam/smish only)" if row.subtype_only else "")
        )


def print_rows(disagree: pd.DataFrame) -> None:
    print("\ndisagreements:")
    for row in disagree.itertuples():
        text = str(row.text).replace("\n", " ")
        note = getattr(row, "reviewer_note", "") or ""
        print(
            f"  {row.review_id} {row.source:<10} stored={row.y3:<6} "
            f"reviewer={row.reviewer_y3:<6} {text[:90]!r}"
            + (f"  note={note}" if note else "")
        )


def evidence_rows(disagree: pd.DataFrame, sources: set[str]) -> list[dict]:
    """Text-free per-row evidence, safe to commit in data_audit.json."""
    out = []
    for row in disagree.itertuples():
        if row.source not in sources:
            continue
        out.append(
            {
                "review_id": row.review_id,
                "row_id": row.row_id,
                "source": row.source,
                "stored": row.y3,
                "reviewer": row.reviewer_y3,
                "note": getattr(row, "reviewer_note", "") or "",
            }
        )
    return out


def log_entry(
    summary: pd.DataFrame,
    old_hash: str,
    new_hash: str,
    n_fix: int,
    accepted: list[dict],
    dropped: list[str],
) -> str:
    lines = [
        "",
        "---",
        "",
        f"## {date.today().isoformat()} — label audit applied; pool hash superseded",
        "",
        "`data/apply_audit.py` folded the reviewed sample back into the modern pool.",
        f"{n_fix} label correction(s) were applied before dedup, so the pool was rebuilt,",
        "and the splits and LOSO folds were re-frozen at the same seed (42).",
        "",
        f"- previous dataset hash: `{old_hash}` — **superseded**",
        f"- new dataset hash: `{new_hash}`",
        "",
        "| source | reviewed | disagree | rate | Wilson 95% |",
        "|---|---|---|---|---|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.source} | {row.reviewed} | {row.disagree} | {row.rate:.1%} | "
            f"{row.wilson_lo:.1%}–{row.wilson_hi:.1%} |"
        )
    for item in accepted:
        lines += [
            "",
            f"**Accepted above the gate: `{item['source']}` at {item['rate']:.1%} "
            f"(Wilson {item['wilson_lo']:.1%}–{item['wilson_hi']:.1%}).**",
            "",
            f"Reason: {item['reason']}",
            "",
            "Its stored labels were left unchanged; the sampled corrections were not applied.",
            "Row-level evidence is in `data/processed/modern/data_audit.json` under",
            "`label_audit.accepted`.",
        ]
    if dropped:
        lines += [
            "",
            "Review rows from sources that are not in `POOL_SOURCES` were skipped: "
            + ", ".join(f"`{src}`" for src in dropped)
            + ". Their overrides were filtered, not applied.",
        ]
        for src in dropped:
            if src in DROPPED_SOURCES:
                lines.append(f"- `{src}` excluded from the pool: {DROPPED_SOURCES[src]}")
    lines += [
        "",
        "Only binary (ham vs malicious) disagreements change `y`; spam-vs-smish",
        "disagreements change `y3` only. Any result computed on the previous hash",
        "is stale and must be re-run.",
        "",
    ]
    return "\n".join(lines)


def parse_accept(values: list[str]) -> dict[str, str]:
    """Parse repeated ``--accept-source NAME:REASON`` into a dict."""
    out: dict[str, str] = {}
    for value in values:
        name, sep, reason = value.partition(":")
        name, reason = name.strip(), reason.strip()
        if not sep or not name or not reason:
            raise SystemExit(f"--accept-source needs NAME:REASON, got {value!r}")
        out[name] = reason
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default=str(DEFAULT_REVIEW))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--apply", action="store_true", help="rebuild the pool with corrections")
    ap.add_argument("--log", default=str(LOG), help="verification log to append to")
    ap.add_argument(
        "--accept-source",
        action="append",
        default=[],
        metavar="NAME:REASON",
        help="proceed above the gate for this source, recording the decision and evidence",
    )
    args = ap.parse_args()

    review_path = Path(args.review)
    if not review_path.exists():
        print(f"no review file at {review_path} — nothing to do")
        return 1

    df = load_review(review_path)
    dropped = sorted(set(df["source"]) - set(POOL_SOURCES))
    if dropped:
        n_dropped = int(df["source"].isin(dropped).sum())
        print(
            f"ignoring {n_dropped} review row(s) from {', '.join(dropped)}: "
            "not in POOL_SOURCES (their overrides are filtered, not applied)"
        )
        df = df[~df["source"].isin(dropped)].reset_index(drop=True)

    accepted_reasons = parse_accept(args.accept_source)
    unknown = sorted(set(accepted_reasons) - set(POOL_SOURCES))
    if unknown:
        print(f"warning: accepted sources not in the pool are ignored: {', '.join(unknown)}")

    summary, disagree = summarise(df)
    print_report(summary)

    unreviewed = int(df["reviewer_label"].isna().sum())
    if unreviewed:
        print(f"\n{unreviewed} of {len(df)} rows have a blank reviewer_label (treated as agree)")

    bad = summary[summary["rate"] > GATE]
    blocking = bad[~bad["source"].isin(accepted_reasons)]
    if not blocking.empty:
        print(
            f"\nSTOP: {', '.join(blocking['source'])} disagree on more than {GATE:.0%} "
            "of reviewed rows. Nothing was written."
        )
        print_rows(disagree[disagree["source"].isin(blocking["source"])])
        print("\nDecide row by row; do not auto-fix a whole source.")
        return 2

    accepted = []
    for row in summary[summary["source"].isin(accepted_reasons)].itertuples():
        accepted.append(
            {
                "source": row.source,
                "reason": accepted_reasons[row.source],
                "reviewed": int(row.reviewed),
                "disagree": int(row.disagree),
                "rate": float(row.rate),
                "wilson_lo": float(row.wilson_lo),
                "wilson_hi": float(row.wilson_hi),
                "evidence": evidence_rows(disagree, {row.source}),
            }
        )
    for item in accepted:
        print(
            f"\nACCEPTED {item['source']} at {item['rate']:.1%} "
            f"[{item['wilson_lo']:.1%}, {item['wilson_hi']:.1%}]: {item['reason']}"
        )
        print(f"  {len(item['evidence'])} disagreement row(s) kept as stored, not overridden")

    if disagree.empty:
        print("\nno disagreements — pool unchanged")
        return 0

    if not args.apply:
        print_rows(disagree)
        print(f"\n{len(disagree)} disagreement(s) below the gate; re-run with --apply to rebuild")
        return 0

    out = Path(args.out)
    audit_path = out / "data_audit.json"
    old_hash = json.loads(audit_path.read_text())["dataset_sha256_16"] if audit_path.exists() else "?"

    overrides = {
        row.row_id: row.reviewer_y3
        for row in disagree.itertuples()
        if row.reviewer_y3 != row.y3 and row.source not in accepted_reasons
    }
    exclusion_path = out / "dedup_exclusions.json"
    dedup_exclusions = (
        json.loads(exclusion_path.read_text()) if exclusion_path.exists() else None
    )
    result = build(label_overrides=overrides, dedup_exclusions=dedup_exclusions)
    result["audit"]["label_audit"] = {
        "review_file": str(review_path),
        "reviewed": int(len(df)),
        "corrections": len(overrides),
        "gate": GATE,
        "per_source": summary.to_dict(orient="records"),
        "accepted": accepted,
        "skipped_review_sources": dropped,
        "supersedes": old_hash,
    }
    write_pool(result, out)
    # Text-free record of the corrections so ``make data`` reproduces this pool.
    (out / "label_overrides.json").write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n")

    new_hash = result["audit"]["dataset_sha256_16"]
    print(f"\npool rebuilt: {old_hash} -> {new_hash}")
    print(f"  rows    {result['audit']['pool_rows']}")
    print("  splits  " + ", ".join(
        f"{k} {v['rows']}" for k, v in result["audit"]["splits"].items()
    ))
    with Path(args.log).open("a") as fh:
        fh.write(log_entry(summary, old_hash, new_hash, len(overrides), accepted, dropped))
    print(f"  log     {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
