"""Record the frozen band for every synthetic demo message (text-free).

Reads the serving bundle once and writes ``reports/results/demo_bands.json``
with the label, band, conformal set and calibrated score only. No message text
is read into the output. Nothing is refit or tuned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from smishlens.demo_examples import EXAMPLES  # noqa: E402
from smishlens.serve import load_bundle, predict_message  # noqa: E402

OUT = ROOT / "reports" / "results" / "demo_bands.json"


def build(artifact: str | None = None) -> dict:
    bundle = load_bundle(artifact) if artifact else load_bundle()
    rows = []
    for row in EXAMPLES:
        result = predict_message(bundle, row["text"])
        rows.append(
            {
                "label": row["label"],
                "band": result.band,
                "conformal_set": result.conformal_set,
                "p_spam": round(float(result.p_spam), 6),
            }
        )
    return {"n_examples": len(rows), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)

    payload = build(args.artifact)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
