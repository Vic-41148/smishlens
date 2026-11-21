"""E17: CPU latency, disk size and RSS for the svm and mmBERT-base arms.

Each arm runs in its own subprocess so the RSS number is not shared. Inputs are
synthetic probe strings generated here; no pool or gold message is read.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from smishlens.calibrate import apply_platt
from smishlens.conformal import conformal_set
from smishlens.pool_experiments import RESULTS, ROOT
from smishlens.serve import DEFAULT_DIR, load_bundle, predict_message

BATCH1_REPS = 30
BATCH32_REPS = 10
MMBERT_BATCH1_REPS = 8
MMBERT_BATCH32_REPS = 4


def _probe_texts(n: int) -> list[str]:
    return [
        (
            f"Synthetic latency probe {i}: your code is {100000 + i}, valid for "
            f"10 minutes. Visit hxxp://example.test/{i} for details."
        )
        for i in range(n)
    ]


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[idx])


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def _dir_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def measure_svm() -> dict:
    bundle = load_bundle(DEFAULT_DIR)
    predict_message(bundle, _probe_texts(1)[0])  # warm up lazy imports/paths
    texts = _probe_texts(BATCH1_REPS)
    peak = _rss_mb()
    single = []
    for text in texts:
        start = time.perf_counter()
        predict_message(bundle, text)
        single.append((time.perf_counter() - start) * 1000)
        peak = max(peak, _rss_mb())

    batch = []
    qs = {int(k): float(v) for k, v in bundle["q_hat"].items()}
    for _ in range(BATCH32_REPS):
        chunk = _probe_texts(32)
        start = time.perf_counter()
        X = bundle["feats"].transform(chunk)
        scores = bundle["clf"].decision_function(X)
        for score in scores:
            p = float(apply_platt([score], bundle["platt"]["a"], bundle["platt"]["b"])[0])
            conformal_set(p, qs)
        batch.append((time.perf_counter() - start) * 1000 / len(chunk))
        peak = max(peak, _rss_mb())

    return {
        "arm": "svm",
        "device": "cpu",
        "batch_1_ms": {"p50": _pct(single, 0.50), "p95": _pct(single, 0.95), "n": len(single)},
        "batch_32_ms_per_message": {
            "p50": _pct(batch, 0.50),
            "p95": _pct(batch, 0.95),
            "n_batches": len(batch),
        },
        "disk_mb": _dir_mb(DEFAULT_DIR),
        "rss_mb": peak,
    }


def measure_mmbert() -> dict:
    import json as _json

    from smishlens.train_mmbert import MMBertScorer

    art = ROOT / "artifacts" / "mmbert" / "e3_base"
    meta = _json.loads((art / "metadata.json").read_text())
    scorer = MMBertScorer.load(art / "mmbert", T=float(meta["calibration"]["T"]), device="cpu")
    scorer.predict_proba(_probe_texts(1))  # warm up the first forward pass
    peak = _rss_mb()
    single = []
    for text in _probe_texts(MMBERT_BATCH1_REPS):
        start = time.perf_counter()
        scorer.predict_proba([text])
        single.append((time.perf_counter() - start) * 1000)
        peak = max(peak, _rss_mb())

    batch = []
    for _ in range(MMBERT_BATCH32_REPS):
        chunk = _probe_texts(32)
        start = time.perf_counter()
        scorer.predict_proba(chunk)
        batch.append((time.perf_counter() - start) * 1000 / len(chunk))
        peak = max(peak, _rss_mb())

    return {
        "arm": "mmbert_base",
        "device": "cpu",
        "batch_1_ms": {"p50": _pct(single, 0.50), "p95": _pct(single, 0.95), "n": len(single)},
        "batch_32_ms_per_message": {
            "p50": _pct(batch, 0.50),
            "p95": _pct(batch, 0.95),
            "n_batches": len(batch),
        },
        "disk_mb": _dir_mb(art),
        "rss_mb": peak,
    }


def _run_one(arm: str) -> dict:
    return measure_svm() if arm == "svm" else measure_mmbert()


def run_all() -> dict:
    rows = []
    for arm in ("svm", "mmbert_base"):
        proc = subprocess.run(
            [sys.executable, "-m", "smishlens.e17_latency", "--one", arm],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
        rows.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    return {
        "scope": "synthetic probe strings, CPU only; each arm in its own subprocess",
        "note": (
            "p50/p95 over repeated calls; batch 32 reports per-message time. "
            "Disk size is the model directory; RSS is peak VmRSS after load."
        ),
        "arms": {row["arm"]: row for row in rows},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", choices=("svm", "mmbert_base"))
    ap.add_argument("--out", default=str(RESULTS / "e17_latency.json"))
    args = ap.parse_args(argv)
    if args.one:
        print(json.dumps(_run_one(args.one)))
        return 0
    payload = run_all()
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    for arm, row in payload["arms"].items():
        print(
            f"{arm}: b1 p50={row['batch_1_ms']['p50']:.2f} p95={row['batch_1_ms']['p95']:.2f} "
            f"b32={row['batch_32_ms_per_message']['p50']:.2f} "
            f"disk={row['disk_mb']:.1f}MB rss={row['rss_mb']:.0f}MB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
