"""Check that a tracked tree matches the public/private deployment split.

Public repository: code, docs and text-free data only. No serving artifact, no
processed text, no raw data.
Private repository: the same tree plus exactly the three serving files.

Run before a push::

    python -m smishlens.public_check            # public mode, inspects main
    python -m smishlens.public_check --ref REF  # public mode, inspects REF
    python -m smishlens.public_check --private  # private mode, current tree

Public mode never inspects the ``private-deploy`` branch: that branch is only
pushed to the ``private`` remote. The check reads ``git`` trees and hashes only;
it never reads message text.
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


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def default_ref() -> str:
    """``main`` when it exists, otherwise ``HEAD``."""
    probe = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--verify", "--quiet", "main"],
        capture_output=True,
        text=True,
    )
    return "main" if probe.returncode == 0 else "HEAD"


def tracked_paths(ref: str | None = None) -> list[str]:
    """Paths in ``ref`` when given, otherwise the working index."""
    if ref:
        return _git("ls-tree", "-r", "--name-only", ref).splitlines()
    return _git("ls-files").splitlines()


def history_hits(ref: str | None = None, path: str = ARTIFACT_DIR) -> list[str]:
    """Commits reachable from ``ref`` that touched the artifact path."""
    args = ["log", "--format=%h"]
    args.append(ref if ref else "--all")
    args += ["--", path]
    return [line for line in _git(*args).splitlines() if line.strip()]


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
    parser.add_argument(
        "--ref",
        action="append",
        default=None,
        help="git ref to inspect in public mode (repeatable); default main",
    )
    args = parser.parse_args(argv)

    if args.private:
        paths = tracked_paths(args.ref[0] if args.ref else None)
        problems = violations(paths, private=True)
        problems += verify_manifest()
        label = "private tree"
    else:
        refs = args.ref or [default_ref()]
        paths = []
        problems = []
        for ref in refs:
            ref_paths = tracked_paths(ref)
            paths = ref_paths
            problems += violations(ref_paths, private=False)
            problems += [
                f"artifact appears in history of {ref}: {commit}"
                for commit in history_hits(ref)
            ]
        label = "public tree at " + ", ".join(refs)

    if problems:
        print(f"FAIL: {len(problems)} problem(s) in the {label}")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(f"OK: {label} matches the deployment split ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
