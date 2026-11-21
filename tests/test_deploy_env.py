"""Deployment harness helpers: pins, probe parsing, artifact and base image."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy_env.py"


def _module():
    spec = importlib.util.spec_from_file_location("deploy_env", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_requirements_are_names_only():
    mod = _module()
    names = mod._direct_requirements()
    assert "scikit-learn" in names
    assert "streamlit" in names
    for name in names:
        assert "==" not in name, name
        assert not name.startswith("-"), name


def test_parse_result_takes_the_last_marker_line():
    mod = _module()
    stdout = 'warning\nRESULT {"a": 1}\nnoise\nRESULT {"a": 2}\n'
    assert mod._parse_result(stdout) == {"a": 2}


def test_parse_result_requires_a_marker():
    mod = _module()
    with pytest.raises(RuntimeError):
        mod._parse_result("nothing here\n")


def test_dockerfile_base_is_the_pinned_slim_image():
    mod = _module()
    assert mod._dockerfile_base() == "python:3.12-slim"


def test_artifact_matches_the_manifest():
    mod = _module()
    if not (mod.ARTIFACT_DIR / "svm_bundle.joblib").exists():
        pytest.skip("serving bundle is private")
    record = mod._artifact()
    assert record["artifact_matches_manifest"] is True
    assert record["artifact_sha256"] == record["manifest_sha256"]
