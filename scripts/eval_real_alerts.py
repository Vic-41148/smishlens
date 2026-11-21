"""Evaluate the frozen bundle on real, locally-held alert messages.

The input is ``data/local/real_alerts.csv`` (columns ``text,label`` with
``ham``/``spam``). It is git-ignored and must never be committed, printed or
logged. This script never refits anything and never changes a threshold: it
scores each row with the frozen svm + Platt + conformal bundle and reports
**aggregates only**.

Reported: row and label counts, band counts, FPR and recall at the frozen
E3 threshold (``tau_raw`` mapped through the stored Platt fit), and mean
calibrated score. No text, no row id, no per-row record is written anywhere.
The run refuses to report when fewer than ``MIN_HAM`` ham rows are present.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from smishlens.calibrate import apply_platt  # noqa: E402
from smishlens.results_summary import num  # noqa: E402
from smishlens.serve import (  # noqa: E402
    BAND_HAM,
    BAND_REVIEW,
    BAND_SCAM,
    DEFAULT_DIR,
    load_bundle,
    predict_message,
)

INPUT = ROOT / "data" / "local" / "real_alerts.csv"
OUT = ROOT / "reports" / "results" / "real_alerts.json"
MIN_HAM = 10
BANDS = (BAND_SCAM, BAND_HAM, BAND_REVIEW)
LABELS = ("ham", "spam")


class InsufficientSample(RuntimeError):
    """Too few ham rows to say anything about a false-positive rate."""


def read_alerts(path: Path = INPUT) -> pd.DataFrame:
    """Load and validate the local evaluation file. Never prints its contents."""
    frame = pd.read_csv(path)
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"real_alerts.csv is missing columns: {sorted(missing)}")
    frame = frame[["text", "label"]].copy()
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    unknown = sorted(set(frame["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"labels must be ham/spam; found {unknown}")
    frame["text"] = frame["text"].fillna("").astype(str)
    return frame


def check_sample(n_ham: int, min_ham: int = MIN_HAM) -> None:
    if n_ham < min_ham:
        raise InsufficientSample(
            f"only {n_ham} ham row(s); need at least {min_ham} to report a "
            "false-positive rate"
        )


def frozen_p_threshold(bundle: dict) -> float:
    """``tau_raw`` mapped through the stored Platt fit. The decision is unchanged."""
    return float(apply_platt([float(bundle["tau_raw"])], bundle["platt"]["a"], bundle["platt"]["b"])[0])


def evaluate(frame: pd.DataFrame, predict, p_threshold: float) -> dict:
    """Aggregate-only report. ``predict(text)`` returns one scored result."""
    bands = {band: 0 for band in BANDS}
    bands_by_label = {label: {band: 0 for band in BANDS} for label in LABELS}
    flagged = {label: 0 for label in LABELS}
    scores = {label: [] for label in LABELS}

    for text, label in zip(frame["text"], frame["label"], strict=True):
        result = predict(text)
        bands[result.band] = bands.get(result.band, 0) + 1
        bands_by_label[label][result.band] += 1
        scores[label].append(float(result.p_spam))
        if result.p_spam >= p_threshold:
            flagged[label] += 1

    n_ham = len(scores["ham"])
    n_spam = len(scores["spam"])
    check_sample(n_ham)
    all_scores = scores["ham"] + scores["spam"]
    return {
        "n": int(len(frame)),
        "per_label": {"ham": n_ham, "spam": n_spam},
        "bands": bands,
        "bands_by_label": bands_by_label,
        "threshold": {
            "rule": "bundle tau_raw, mapped through the stored Platt fit",
            "p_spam": round(float(p_threshold), 9),
        },
        "flagged_at_frozen_threshold": flagged,
        "fpr_at_frozen_threshold": round(flagged["ham"] / n_ham, 6) if n_ham else None,
        "recall_at_frozen_threshold": round(flagged["spam"] / n_spam, 6) if n_spam else None,
        "mean_calibrated_score": {
            "all": round(sum(all_scores) / len(all_scores), 6) if all_scores else None,
            "ham": round(sum(scores["ham"]) / n_ham, 6) if n_ham else None,
            "spam": round(sum(scores["spam"]) / n_spam, 6) if n_spam else None,
        },
    }


def run(input_path: Path = INPUT, artifact_dir: Path = DEFAULT_DIR) -> dict:
    frame = read_alerts(input_path)
    n_ham = int((frame["label"] == "ham").sum())
    check_sample(n_ham)
    bundle = load_bundle(artifact_dir)
    p_threshold = frozen_p_threshold(bundle)
    report = evaluate(frame, lambda text: predict_message(bundle, text), p_threshold)
    report.update(
        {
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "input": "data/local/real_alerts.csv",
            "artifact_sha256": bundle.get("meta", {}).get("artifact_sha256"),
            "min_ham_required": MIN_HAM,
            "evaluation_only": True,
            "refit": False,
            "threshold_changed": False,
        }
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--artifact", default=str(DEFAULT_DIR))
    args = parser.parse_args(argv)

    if not Path(args.input).exists():
        print(f"real alerts: input file not present at {args.input}; nothing to evaluate")
        return 1

    try:
        report = run(Path(args.input), Path(args.artifact))
    except InsufficientSample as exc:
        print(f"real alerts: refusing to report: {exc}")
        return 2
    except ValueError as exc:
        print(f"real alerts: cannot use {args.input}: {exc}")
        return 3
    except FileNotFoundError as exc:
        print(f"real alerts: {exc}")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(
        f"n={report['n']} ham={report['per_label']['ham']} spam={report['per_label']['spam']} "
        f"FPR={num(report['fpr_at_frozen_threshold'])} "
        f"recall={num(report['recall_at_frozen_threshold'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
