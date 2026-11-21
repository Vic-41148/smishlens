"""Pre-push hook logic, exercised in throwaway repositories.

The hook is copied from ``deploy/pre-push`` into a temp repo's ``.git/hooks``.
No real remote is contacted and the real repository's remotes are never read
or written.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / "deploy" / "pre-push"
SRC = Path(__file__).resolve().parents[1] / "src"
ZERO = "0" * 40

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=GIT_ENV,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "README.md").write_text("text-free fixture\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "fixture")
    return repo


def _install_hook(repo: Path) -> Path:
    hook = repo / ".git" / "hooks" / "pre-push"
    hook.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, hook)
    hook.chmod(0o755)
    return hook


def _run_hook(repo: Path, remote: str, lines: list[str]) -> subprocess.CompletedProcess:
    hook = _install_hook(repo)
    env = {
        **GIT_ENV,
        "SMISHLENS_REPO_ROOT": str(repo),
        # The hook prepends <repo>/src, which does not exist here; the real
        # package is found through this entry.
        "PYTHONPATH": str(SRC),
    }
    return subprocess.run(
        ["sh", str(hook), remote, "ssh://example.invalid/repo.git"],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        cwd=repo,
        env=env,
    )


def _line(local_ref: str, sha: str, remote_ref: str | None = None) -> str:
    return f"{local_ref} {sha} {remote_ref or local_ref} {ZERO}"


def test_refuses_private_deploy_on_a_non_private_remote(tmp_path):
    repo = _repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    result = _run_hook(repo, "origin", [_line("refs/heads/private-deploy", sha)])
    assert result.returncode != 0
    assert "private-deploy" in result.stderr
    assert "private" in result.stderr


def test_refuses_artifact_history_on_origin(tmp_path):
    repo = _repo(tmp_path)
    art = repo / "artifacts" / "serve"
    art.mkdir(parents=True)
    (art / "svm_bundle.joblib").write_bytes(b"fixture")
    _git(repo, "add", "artifacts/serve/svm_bundle.joblib")
    _git(repo, "commit", "-q", "-m", "artifact")
    sha = _git(repo, "rev-parse", "HEAD")
    result = _run_hook(repo, "origin", [_line("refs/heads/main", sha)])
    assert result.returncode != 0
    assert "artifacts/serve" in result.stderr


def test_clean_ref_passes_public_check(tmp_path):
    repo = _repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    result = _run_hook(repo, "origin", [_line("refs/heads/main", sha)])
    assert result.returncode == 0, result.stderr + result.stdout
    assert "OK:" in result.stdout


def test_private_remote_runs_private_mode(tmp_path):
    repo = _repo(tmp_path)
    _git(repo, "switch", "-q", "-c", "private-deploy")
    art = repo / "artifacts" / "serve"
    art.mkdir(parents=True)
    names = ["svm_bundle.joblib", "metadata.json"]
    for name in names:
        (art / name).write_bytes(f"fixture {name}".encode())
    manifest = "".join(
        f"{hashlib.sha256((art / name).read_bytes()).hexdigest()}  {name}\n" for name in names
    )
    (art / "MANIFEST.sha256").write_text(manifest)
    _git(repo, "add", "artifacts/serve")
    _git(repo, "commit", "-q", "-m", "artifact")
    sha = _git(repo, "rev-parse", "HEAD")
    result = _run_hook(repo, "private", [_line("refs/heads/private-deploy", sha)])
    assert result.returncode == 0, result.stderr + result.stdout
    assert "private tree" in result.stdout


def test_private_remote_public_ref_still_uses_public_mode(tmp_path):
    repo = _repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    result = _run_hook(repo, "private", [_line("refs/heads/main", sha)])
    assert result.returncode == 0, result.stderr + result.stdout
    assert "public tree" in result.stdout


@pytest.mark.parametrize("remote", ["origin", "private"])
def test_deletions_are_ignored(tmp_path, remote):
    repo = _repo(tmp_path)
    result = _run_hook(repo, remote, [_line("(delete)", ZERO)])
    assert result.returncode == 0


@pytest.mark.parametrize(
    "path",
    ["data/local/real_alerts.csv", "reports/results/real_alerts.json"],
)
def test_refuses_local_real_alerts_on_origin(tmp_path, path):
    repo = _repo(tmp_path)
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("text-free fixture\n")
    _git(repo, "add", "-f", path)
    _git(repo, "commit", "-q", "-m", "local set")
    sha = _git(repo, "rev-parse", "HEAD")
    result = _run_hook(repo, "origin", [_line("refs/heads/main", sha)])
    assert result.returncode != 0
    assert "forbidden" in result.stdout
