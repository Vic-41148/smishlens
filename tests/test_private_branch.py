"""Public mode inspects main (or --ref); private mode inspects the checked-out tree.

These tests build throwaway git repositories. No real remote is contacted.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from smishlens import public_check

ARTIFACT_FILES = ("svm_bundle.joblib", "metadata.json", "MANIFEST.sha256")
GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args],
        capture_output=True,
        text=True,
        check=True,
        env=GIT_ENV,
    )
    return result.stdout.strip()


def _add_artifact(repo: Path) -> None:
    directory = repo / "artifacts" / "serve"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "svm_bundle.joblib").write_bytes(b"fixture bundle")
    (directory / "metadata.json").write_text("{}\n")
    lines = []
    for name in ("svm_bundle.joblib", "metadata.json"):
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    (directory / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "app.py").write_text("print('fixture')\n")
    (repo / ".gitignore").write_text("artifacts/\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "public tree")

    _git(repo, "checkout", "-q", "-b", "private-deploy")
    _add_artifact(repo)
    (repo / ".gitignore").write_text("artifacts/\n!artifacts/serve/\n")
    _git(repo, "add", "-f", "artifacts/serve", ".gitignore")
    _git(repo, "commit", "-q", "-m", "private tree")
    _git(repo, "checkout", "-q", "main")

    monkeypatch.setenv("SMISHLENS_REPO_ROOT", str(repo))
    return repo


def test_public_mode_inspects_main_and_ignores_private_deploy(repo, capsys):
    assert public_check.main([]) == 0
    out = capsys.readouterr().out
    assert "public tree at main" in out


def test_public_mode_can_inspect_an_explicit_private_ref(repo, capsys):
    assert public_check.main(["--ref", "private-deploy"]) == 1
    assert "artifact file not allowed here" in capsys.readouterr().out


def test_private_mode_passes_on_the_checked_out_private_tree(repo):
    _git(repo, "checkout", "-q", "private-deploy")
    assert public_check.main(["--private"]) == 0


def test_private_mode_fails_on_the_public_tree(repo, capsys):
    assert public_check.main(["--private"]) == 1
    assert "missing artifact file" in capsys.readouterr().out


def test_private_mode_detects_a_tampered_bundle(repo):
    _git(repo, "checkout", "-q", "private-deploy")
    (repo / "artifacts" / "serve" / "svm_bundle.joblib").write_bytes(b"tampered")
    assert public_check.main(["--private"]) == 1
