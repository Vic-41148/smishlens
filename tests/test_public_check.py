"""Deployment split checks: tracked paths and manifest hashes."""

from __future__ import annotations

import hashlib

from smishlens.public_check import ARTIFACT_FILES, verify_manifest, violations


def test_public_tree_rejects_the_serving_bundle():
    problems = violations(["artifacts/serve/svm_bundle.joblib"])
    assert problems


def test_public_tree_rejects_processed_splits():
    for path in (
        "data/processed/modern/train.csv",
        "data/processed/modern/cal.csv",
        "data/processed/modern/test.csv",
    ):
        assert violations([path]), path


def test_public_tree_rejects_raw_and_review_data():
    assert violations(["data/raw/uci/SMSSpamCollection"])
    assert violations(["data/processed/review/review_sample.csv"])
    assert violations(["session-notes.md"])


def test_public_tree_rejects_local_real_alerts():
    assert violations(["data/local/real_alerts.csv"])
    assert violations(["data/local/real_alerts.json"])
    assert violations(["reports/results/real_alerts.json"])


def test_public_tree_accepts_code_and_docs():
    assert not violations(
        [
            "src/smishlens/serve.py",
            "app/app.py",
            "docs/RESULTS_SUMMARY.md",
            "data/processed/modern/split_manifest.csv",
            "reports/results/e3_pool.json",
        ]
    )


def test_private_tree_requires_all_three_files():
    problems = violations(["src/smishlens/serve.py"], private=True)
    assert len(problems) == len(ARTIFACT_FILES)


def test_private_tree_accepts_the_three_files():
    assert not violations(list(ARTIFACT_FILES) + ["src/smishlens/serve.py"], private=True)


def test_private_tree_rejects_extra_artifact_files():
    problems = violations(list(ARTIFACT_FILES) + ["artifacts/serve/other.joblib"], private=True)
    assert any("other.joblib" in problem for problem in problems)


def _write_manifest(directory, names):
    lines = []
    for name in names:
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    (directory / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def test_manifest_round_trip(tmp_path):
    (tmp_path / "svm_bundle.joblib").write_bytes(b"bundle")
    (tmp_path / "metadata.json").write_text("{}")
    _write_manifest(tmp_path, ["svm_bundle.joblib", "metadata.json"])
    assert not verify_manifest(tmp_path)


def test_manifest_detects_tampering(tmp_path):
    (tmp_path / "svm_bundle.joblib").write_bytes(b"bundle")
    (tmp_path / "metadata.json").write_text("{}")
    _write_manifest(tmp_path, ["svm_bundle.joblib", "metadata.json"])
    (tmp_path / "svm_bundle.joblib").write_bytes(b"tampered")
    problems = verify_manifest(tmp_path)
    assert any("svm_bundle.joblib" in problem for problem in problems)
