"""Generated deployment-environment section: values come only from the JSON."""

from __future__ import annotations

import json

from smishlens import deploy_doc


def _write(tmp_path, payload: dict):
    path = tmp_path / "deploy_env.json"
    path.write_text(json.dumps(payload))
    return path


def test_missing_json_says_not_yet_run(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy_doc, "DEPLOY_ENV", tmp_path / "deploy_env.json")
    lines = deploy_doc.deploy_env_lines()
    assert "Not yet run" in "\n".join(lines)
    assert "make deploy-env" in "\n".join(lines)


def test_values_are_read_from_the_json(tmp_path, monkeypatch):
    payload = {
        "generated_utc": "2026-11-22T10:00:00+00:00",
        "requirements": "requirements.txt",
        "python_version": "3.12.14",
        "packages": {"scikit-learn": "1.9.1", "streamlit": "1.67.0"},
        "artifact_sha256": "eeca262df368dcabd58d803038a876ca510c549b9947332a3cb2d88b96a28843",
        "artifact_matches_manifest": True,
        "app_import_ok": True,
        "bundle_loaded": True,
        "apptest_ok": True,
        "csvs_absent": True,
        "csv_restore_verified": True,
        "cross_version": {
            "reference_python": "3.14.10",
            "match": True,
            "max_abs_diff": 0.0,
            "tolerance": 1e-9,
        },
        "docker": {
            "tested": True,
            "base_image": "python:3.12-slim",
            "image_size_mb": 812.3,
            "health_ok": True,
            "no_pool_csvs": True,
            "image_removed": True,
        },
    }
    monkeypatch.setattr(deploy_doc, "DEPLOY_ENV", _write(tmp_path, payload))
    text = "\n".join(deploy_doc.deploy_env_lines())

    assert "3.12.14" in text
    assert "`scikit-learn==1.9.1`" in text
    assert "eeca262df368dcabd58d803038a876ca510c549b9947332a3cb2d88b96a28843" in text
    assert "812.3 MB" in text
    assert "python:3.12-slim" in text
    assert "3.14.10" in text
    assert "0.000000000000" in text


def test_failed_checks_render_as_no(tmp_path, monkeypatch):
    payload = {
        "requirements": "requirements.txt",
        "python_version": "3.12.14",
        "packages": {},
        "artifact_matches_manifest": False,
        "apptest_ok": False,
        "csv_restore_verified": False,
        "cross_version": {"reference_python": "3.14.10", "match": False, "max_abs_diff": 0.5},
        "docker": {"tested": False, "note": "docker daemon unavailable"},
    }
    monkeypatch.setattr(deploy_doc, "DEPLOY_ENV", _write(tmp_path, payload))
    text = "\n".join(deploy_doc.deploy_env_lines())
    assert "**no**" in text
    assert "docker daemon unavailable" in text


def test_section_is_wired_into_the_document():
    assert "## Deployment-environment check" in deploy_doc.deployment()
