"""Check that a tracked tree matches the public/private deployment split.

Public repository: code, docs and text-free data only. No serving artifact, no
processed text, no raw data.
Private repository: the same tree plus exactly the three serving files.

Run before a push::

    python -m smishlens.public_check            # public repo
    python -m smishlens.public_check --private  # private repo

The check reads ``git ls-files`` and hashes only; it never reads message text.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = "artifacts/serve"
ARTIFACT_FILES = (
    "artifacts/serve/svm_bundle.joblib",
    "artifacts/serve/metadata.json",
    "artifacts/serve/MANIFEST.sha256",
)
FORBIDDEN_PREFIXES = (
    "data/raw/",
    "data/processed/review/",
)
FORBIDDEN_PATHS = (
    "data/processed/modern/train.csv",
    "data/processed/modern/cal.csv",
    "data/processed/modern/test.csv",
)
FORBIDDEN_SUFFIXES = (".joblib", ".pt", ".bin", ".safetensors", ".onnx")
ALLOWED_PATHS = ("data/raw/.gitkeep", "artifacts/.gitkeep")


def violations(paths: list[str], private: bool = False) -> list[str]:
    """Return human-readable problems for a list of tracked paths."""
    paths = list(paths)
    allowed = set(ARTIFACT_FILES) if private else set()
    problems: list[str] = []
    for path in paths:
        if path in allowed or path in ALLOWED_PATHS:
            continue
        if path.startswith(ARTIFACT_DIR + "/"):
            problems.append(f"artifact file not allowed here: {path}")
            continue
        if path in FORBIDDEN_PATHS or path.startswith(FORBIDDEN_PREFIXES):
            problems.append(f"forbidden path: {path}")
            continue
        if path.endswith(FORBIDDEN_SUFFIXES):
            problems.append(f"forbidden file type: {path}")
            continue
        name = path.rsplit("/", 1)[-1]
        if name == "review_sample.csv" or (
            name.startswith("session-") and name.endswith(".md")
        ):
            problems.append(f"forbidden file: {path}")
    if private:
        for required in ARTIFACT_FILES:
            if required not in paths:
                problems.append(f"missing artifact file: {required}")
    return problems


def tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def history_hits(path: str = ARTIFACT_DIR) -> list[str]:
    """Commits that touched the artifact path, across all local refs."""
    result = subprocess.run(
        ["git", "-C", str(ROOT), "log", "--all", "--format=%h", "--", path],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def verify_manifest(artifact_dir: Path | None = None) -> list[str]:
    """Check every ``sha256  name`` line in MANIFEST.sha256."""
    directory = artifact_dir or ROOT / ARTIFACT_DIR
    manifest = directory / "MANIFEST.sha256"
    if not manifest.exists():
        return [f"missing manifest: {manifest}"]
    problems: list[str] = []
    for line in manifest.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            problems.append(f"malformed manifest line: {line!r}")
            continue
        digest, name = parts
        target = directory / name
        if not target.exists():
            problems.append(f"manifest lists a missing file: {name}")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            problems.append(f"hash mismatch: {name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args(argv)

    paths = tracked_paths()
    problems = violations(paths, private=args.private)
    if not args.private:
        for commit in history_hits():
            problems.append(f"artifact appears in history: {commit}")
    else:
        problems += verify_manifest()

    if problems:
        print(f"FAIL: {len(problems)} problem(s) in the tracked tree")
        for problem in problems:
            print(f"- {problem}")
        return 1
    mode = "private" if args.private else "public"
    print(f"OK: {mode} tree matches the deployment split ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
