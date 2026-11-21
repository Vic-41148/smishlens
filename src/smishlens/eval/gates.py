"""Fail the build when a metric regresses.

``floors.json`` maps a metric name to ``{"min": x}`` or ``{"max": y}``. Set the
numbers slightly below your own first measured results: a floor is a regression
guard, not an aspiration, and never comes from a paper.
"""

from __future__ import annotations

import argparse
import json
import sys


def check(metrics: dict, floors: dict) -> list[str]:
    """Return one human-readable failure per violated or missing metric."""
    bad: list[str] = []
    for key, rule in floors.items():
        value = metrics.get(key)
        if value is None:
            bad.append(f"{key}: missing from metrics")
            continue
        if value != value:  # NaN
            bad.append(f"{key}: NaN")
            continue
        if "min" in rule and value < rule["min"]:
            bad.append(f"{key}: {value:.4f} < min {rule['min']}")
        elif "max" in rule and value > rule["max"]:
            bad.append(f"{key}: {value:.4f} > max {rule['max']}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check metrics against floors.json")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--floors", required=True)
    args = ap.parse_args(argv)
    with open(args.metrics, encoding="utf-8") as fh:
        metrics = json.load(fh)
    with open(args.floors, encoding="utf-8") as fh:
        floors = json.load(fh)
    bad = check(metrics, floors)
    print("\n".join(bad) or "all gates passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
