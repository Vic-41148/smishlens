"""Pool-svm inference for the hosted app: one calibrated score, one set, one band.

The bundle is fitted from the pool TRAIN split and calibrated on ``cal_a`` only,
the same recipe as E13/E14. No test row and no gold row enters it. Bands come
only from the frozen class-conditional conformal rule; nothing is tuned here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import sklearn
from sklearn.model_selection import train_test_split

from smishlens.calibrate import apply_platt, fit_platt
from smishlens.conformal import conformal_set, fit_mondrian
from smishlens.normalise import URL
from smishlens.pool_experiments import (
    RESULTS,
    ROOT,
    _est,
    build_fold_matrices,
    load_pool,
    transform_with,
)
from smishlens.predict import make_manifest, verify_manifest

MAX_CHARS = 5000
ALPHA = 0.05
DEFAULT_DIR = ROOT / "artifacts" / "serve"
BUNDLE_NAME = "svm_bundle.joblib"
E3_CLASSICAL = RESULTS / "e3_pool.json"

BAND_SCAM = "likely scam"
BAND_HAM = "looks ordinary"
BAND_REVIEW = "needs review"

MISSING_BUNDLE_MESSAGE = (
    "Model bundle not distributed in this repo. Build it with `make build-serve`; "
    "that needs your own licensed copy of the pool data."
)


class BundleVersionError(RuntimeError):
    """The bundle was built with a different scikit-learn version."""


@dataclass(frozen=True)
class ServeResult:
    """What every surface shows. No hard verdict is implied by the band."""

    band: str
    conformal_set: list[int]
    p_spam: float
    p_ham: float
    urls: list[str]
    n_chars: int
    n_words: int
    n_urls: int
    digit_ratio: float
    truncated: bool
    latency_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


def band_for_set(prediction_set: list[int]) -> str:
    """Three display bands from the frozen conformal set only."""
    if prediction_set == [1]:
        return BAND_SCAM
    if prediction_set == [0]:
        return BAND_HAM
    return BAND_REVIEW


def extract_urls(text: str, limit: int = 5) -> list[str]:
    """URLs as written, deduplicated in first-seen order."""
    seen: list[str] = []
    for match in URL.findall(text):
        if match not in seen:
            seen.append(match)
        if len(seen) >= limit:
            break
    return seen


def message_stats(text: str) -> dict:
    words = text.split()
    n = max(len(text), 1)
    return {
        "n_chars": len(text),
        "n_words": len(words),
        "n_urls": len(extract_urls(text, limit=100)),
        "digit_ratio": sum(c.isdigit() for c in text) / n,
    }


def predict_message(bundle: dict, text: object, max_chars: int = MAX_CHARS) -> ServeResult:
    """Score one message. Empty input is scored as empty, never raises."""
    start = time.perf_counter()
    raw_text = "" if text is None else str(text)
    truncated = len(raw_text) > max_chars
    clipped = raw_text[:max_chars]
    feats = bundle["feats"].transform([clipped])
    score = float(bundle["clf"].decision_function(feats)[0])
    p_spam = float(apply_platt([score], bundle["platt"]["a"], bundle["platt"]["b"])[0])
    qs = {int(k): float(v) for k, v in bundle["q_hat"].items()}
    prediction_set = conformal_set(p_spam, qs)
    stats = message_stats(clipped)
    return ServeResult(
        band=band_for_set(prediction_set),
        conformal_set=prediction_set,
        p_spam=p_spam,
        p_ham=1.0 - p_spam,
        urls=extract_urls(clipped),
        n_chars=stats["n_chars"],
        n_words=stats["n_words"],
        n_urls=stats["n_urls"],
        digit_ratio=stats["digit_ratio"],
        truncated=truncated,
        latency_ms=(time.perf_counter() - start) * 1000,
    )


def _pool_manifest_hash() -> str:
    path = ROOT / "data" / "processed" / "modern" / "split_manifest.csv"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _tau_raw() -> float:
    data = json.loads(E3_CLASSICAL.read_text())
    return float(next(r["tau"] for r in data["rows"] if r.get("model") == "svm" and "tau" in r))


def build_bundle(frame=None) -> tuple[dict, dict]:
    """Fit train -> Platt on cal_a -> Mondrian q_hat on cal_a. Nothing else."""
    frame = load_pool() if frame is None else frame
    train = frame[frame["split"] == "train"]
    cal = frame[frame["split"] == "cal"]
    cal_a, _ = train_test_split(cal, test_size=0.5, stratify=cal["y"], random_state=42)

    mats = build_fold_matrices(train)
    clf = _est("svm").fit(mats.X, mats.y)
    s_a = clf.decision_function(transform_with(mats, cal_a["text"]))
    y_a = cal_a["y"].to_numpy()
    a, b = fit_platt(s_a, y_a)
    p_a = apply_platt(s_a, a, b)
    qs = fit_mondrian(p_a, y_a, alpha=ALPHA)

    metadata = {
        "model_name": "pool_svm_platt_mondrian",
        "model_version": "1.0.0",
        "sklearn_version": sklearn.__version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "recipe": (
            "LinearSVC(C=0.5, class_weight=balanced, dual=auto, random_state=42) "
            "on the pool train split; Platt on cal_a decision scores; "
            "class-conditional q_hat on the same cal_a probabilities"
        ),
        "n_train": int(len(train)),
        "n_cal": int(len(cal)),
        "n_cal_a": int(len(cal_a)),
        "alpha": ALPHA,
        "platt": {"a": a, "b": b},
        "q_hat": {str(k): float(v) for k, v in qs.items()},
        "tau_raw": _tau_raw(),
        "pool_manifest_sha256_16": _pool_manifest_hash(),
        "test_or_gold_used": False,
    }
    bundle = {
        "feats": mats.feats,
        "clf": clf,
        "platt": {"a": a, "b": b},
        "q_hat": {str(k): float(v) for k, v in qs.items()},
        "tau_raw": metadata["tau_raw"],
        "meta": metadata,
    }
    return bundle, metadata


def save_bundle(directory: Path, bundle: dict, metadata: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / BUNDLE_NAME
    joblib.dump(bundle, path)
    digest = artifact_sha256(path)
    metadata = {
        "sklearn_version": sklearn.__version__,
        **metadata,
        "artifact_sha256": digest,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    make_manifest(directory)
    return path


def check_sklearn_version(meta: dict) -> None:
    """Refuse a pickle built by a different scikit-learn version."""
    built = meta.get("sklearn_version")
    if not built:
        raise BundleVersionError(
            "bundle records no scikit-learn version; rebuild it with `make build-serve`"
        )
    if built != sklearn.__version__:
        raise BundleVersionError(
            f"bundle was built with scikit-learn {built}, but this environment has "
            f"{sklearn.__version__}; rebuild with `make build-serve` and the pinned "
            "requirements"
        )


def load_bundle(directory: str | Path = DEFAULT_DIR) -> dict:
    """Verify the manifest before unpickling, then return the bundle."""
    directory = Path(directory)
    if not (directory / BUNDLE_NAME).exists():
        raise FileNotFoundError(MISSING_BUNDLE_MESSAGE)
    verify_manifest(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    if meta.get("test_or_gold_used"):
        raise RuntimeError("refusing a bundle that used test or gold rows")
    check_sklearn_version(meta)
    bundle = joblib.load(directory / BUNDLE_NAME)
    bundle["meta"] = meta
    return bundle


def artifact_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_DIR))
    args = ap.parse_args(argv)
    if not args.build:
        ap.error("pass --build")
    bundle, metadata = build_bundle()
    path = save_bundle(Path(args.out), bundle, metadata)
    print(f"wrote {path}")
    print(f"artifact_sha256 {artifact_sha256(path)}")
    print(f"q_hat {metadata['q_hat']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
