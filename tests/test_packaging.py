"""Packaging rules: pinned requirements, app-minimal split, Docker ignore list."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*==[0-9][^\s]*$")
REMOVED_REQUIREMENTS = "requirements" + "-app.txt"
TEXT_FILES = (
    "README.md",
    "MODEL_CARD.md",
    "docs/DEPLOYMENT.md",
    "docs/DEMO_SCRIPT.md",
    "Makefile",
    "Dockerfile",
    "src/smishlens/docs_gen.py",
    "src/smishlens/deploy_doc.py",
)


def _lines(path: Path) -> list[str]:
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in _lines(path):
        if line.startswith("-"):
            continue
        name, version = line.split("==", 1)
        pins[name] = version
    return pins


def test_runtime_requirements_are_exactly_pinned():
    lines = _lines(ROOT / "requirements.txt")
    assert lines, "requirements.txt is empty"
    for line in lines:
        assert PIN.match(line), f"not an exact pin: {line!r}"


def test_runtime_requirements_are_app_minimal():
    pins = _pins(ROOT / "requirements.txt")
    # Everything the app/serve import chain needs, and nothing from the pipeline.
    for name in ("joblib", "numpy", "pandas", "scikit-learn", "scipy", "streamlit"):
        assert name in pins, name
    for name in ("pytest", "ruff", "matplotlib", "fastapi", "uvicorn", "httpx"):
        assert name not in pins, f"{name} belongs in requirements-dev.txt"


def test_dev_requirements_extend_the_runtime_pins():
    lines = _lines(ROOT / "requirements-dev.txt")
    assert lines[0] == "-r requirements.txt"
    for line in lines[1:]:
        assert PIN.match(line), f"not an exact pin: {line!r}"


def test_bundle_pins_match_metadata():
    meta_path = ROOT / "artifacts" / "serve" / "metadata.json"
    if not meta_path.exists():
        pytest.skip("serving bundle is private; nothing to compare here")
    meta = json.loads(meta_path.read_text())
    pins = _pins(ROOT / "requirements.txt")
    assert pins["scikit-learn"] == meta["sklearn_version"]
    assert pins["joblib"] == meta["joblib_version"]


def test_old_requirements_file_is_gone():
    assert not (ROOT / REMOVED_REQUIREMENTS).exists()
    for name in TEXT_FILES:
        path = ROOT / name
        assert REMOVED_REQUIREMENTS not in path.read_text(), name


def test_dockerignore_excludes_data_and_generated_artifacts():
    patterns = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    for required in (
        "data",
        "data/local",
        "models",
        "checkpoints",
        "reports",
        "session-*.md",
    ):
        assert required in patterns, f".dockerignore must exclude {required!r}"


def test_dockerignore_ships_only_the_serving_bundle():
    text = (ROOT / ".dockerignore").read_text()
    assert "!artifacts/serve/svm_bundle.joblib" in text
    assert "!artifacts/serve/metadata.json" in text
    assert "!artifacts/serve/MANIFEST.sha256" in text
