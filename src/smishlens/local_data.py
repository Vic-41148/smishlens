"""Guard and manifest for the text-bearing splits that stay out of the repo.

``data/processed/uci/{train,cal,test}.csv`` and ``data/gold_holdout*.csv``
are never committed. Every entry point that needs them calls ``require_*``
first, so a fresh clone fails with one clear instruction instead of a
``FileNotFoundError`` raised deep inside pandas.

The manifest holds SHA-256 digests and row counts only. It contains no
message text, so it is safe to commit.

Usage::

    python -m smishlens.local_data --write     # regenerate the manifest
    python -m smishlens.local_data --verify    # check the local split
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UCI_DIR = ROOT / "data" / "processed" / "uci"
UCI_SPLITS = ("train", "cal", "test")
MANIFEST_NAME = "SPLIT_MANIFEST.sha256"
GOLD_FILES = ("gold_holdout.csv", "gold_holdout_v2.csv")
DATA_DIR = ROOT / "data"

SETUP_HINT = (
    "run `make setup` (or `python data/download_data.py --only uci` then "
    "`python data/build_uci_anchor.py`) to rebuild them"
)
GOLD_HINT = (
    "the gold holdout is local-only and is not distributed; restore it from "
    "your own backup into data/"
)


def uci_paths(base: Path | None = None) -> list[Path]:
    """The three anchor splits, in a fixed order."""
    directory = UCI_DIR if base is None else Path(base)
    return [directory / f"{name}.csv" for name in UCI_SPLITS]


def gold_paths(base: Path | None = None) -> list[Path]:
    """The gold holdout files, in a fixed order."""
    directory = DATA_DIR if base is None else Path(base)
    return [directory / name for name in GOLD_FILES]


def require_file(path: Path, hint: str = SETUP_HINT) -> None:
    """Exit with one instruction when a caller-supplied path is absent."""
    if not Path(path).exists():
        raise SystemExit(f"missing {path}. Please {hint}.")


def missing(paths: list[Path]) -> list[str]:
    """Repository-relative names of the paths that do not exist."""
    absent = []
    for path in paths:
        if path.exists():
            continue
        absent.append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
    return absent


def require_uci(base: Path | None = None) -> None:
    """Exit with one instruction when the anchor splits are absent."""
    absent = missing(uci_paths(base))
    if absent:
        raise SystemExit(
            "missing UCI anchor splits: " + ", ".join(absent) + ". Please " + SETUP_HINT + "."
        )


def require_gold(base: Path | None = None) -> None:
    """Exit with one instruction when the gold holdout is absent."""
    absent = missing(gold_paths(base))
    if absent:
        raise SystemExit("missing gold holdout: " + ", ".join(absent) + ". " + GOLD_HINT + ".")


def sha256(path: Path) -> str:
    """Hex digest of a file, read in chunks so a large CSV stays out of memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def counts(path: Path) -> tuple[int, int, int]:
    """``(rows, spam, ham)`` from the ``y`` column only. No text is read."""
    rows = spam = 0
    with Path(path).open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            rows += 1
            if str(record.get("y", "")).strip() in {"1", "1.0"}:
                spam += 1
    return rows, spam, rows - spam


def manifest_lines(base: Path | None = None) -> list[str]:
    """One ``sha256  name  rows  spam  ham`` line per anchor split."""
    lines = []
    for path in uci_paths(base):
        rows, spam, ham = counts(path)
        lines.append(f"{sha256(path)}  {path.name}  {rows}  {spam}  {ham}")
    return lines


def write_manifest(base: Path | None = None) -> Path:
    """Write the manifest next to the splits and return its path."""
    directory = UCI_DIR if base is None else Path(base)
    target = directory / MANIFEST_NAME
    header = "# UCI anchor split: sha256  file  rows  spam  ham (text-free)"
    target.write_text("\n".join([header, *manifest_lines(base)]) + "\n")
    return target


def verify_manifest(base: Path | None = None) -> list[str]:
    """Return problems; an empty list means the local split matches the manifest."""
    directory = UCI_DIR if base is None else Path(base)
    manifest = directory / MANIFEST_NAME
    if not manifest.exists():
        return [f"missing manifest: {manifest}"]
    expected = {}
    for line in manifest.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            return [f"malformed manifest line: {line!r}"]
        digest, name, rows, spam, ham = parts
        expected[name] = (digest, int(rows), int(spam), int(ham))
    problems = []
    for path in uci_paths(directory):
        if path.name not in expected:
            problems.append(f"manifest does not list {path.name}")
            continue
        digest, rows, spam, ham = expected[path.name]
        if not path.exists():
            problems.append(f"missing split: {path.name}")
            continue
        if sha256(path) != digest:
            problems.append(f"hash mismatch: {path.name}")
            continue
        if counts(path) != (rows, spam, ham):
            problems.append(f"row counts differ: {path.name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UCI anchor split manifest")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="regenerate the manifest")
    group.add_argument("--verify", action="store_true", help="check the local split")
    args = parser.parse_args(argv)

    require_uci()
    if args.write:
        target = write_manifest()
        print(f"wrote {target.relative_to(ROOT)}")
        for line in manifest_lines():
            print(f"  {line}")
        return 0

    problems = verify_manifest()
    if problems:
        print(f"FAIL: {len(problems)} problem(s)")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(f"OK: {len(UCI_SPLITS)} splits match {MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
