"""Pre-push hook dispatch, exercised in throwaway repositories.

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
PRIVATE_URL = "ssh://example.invalid/private.git"
ORIGIN_URL = "ssh://example.invalid/public.git"

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


def _run_hook(repo: Path, remote: str, lines: list[str], url: str = ORIGIN_URL):
    hook = _install_hook(repo)
    env = {
        **GIT_ENV,
        "SMISHLENS_REPO_ROOT": str(repo),
        # The hook prepends <repo>/src, which does not exist here; the real
        # package is found through this entry.
        "PYTHONPATH": str(SRC),
    }
    return subprocess.run(
        ["sh", str(hook), remote, url],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        cwd=repo,
        env=env,
    )


def _line(local_ref: str, sha: str, remote_ref: str | None = None, remote_sha: str = ZERO) -> str:
    return f"{local_ref} {sha} {remote_ref or local_ref} {remote_sha}"


def _add_private_remote(repo: Path, url: str = PRIVATE_URL) -> None:
    _git(repo, "remote", "add", "private", url)


def _commit_artifacts(repo: Path) -> str:
    """Commit the three serving files on a private-deploy branch and return its sha."""
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
    _git(repo, "add", "-f", "artifacts/serve")
    _git(repo, "commit", "-q", "-m", "artifact")
    return _git(repo, "rev-parse", "HEAD")


def _orphan_sha(repo: Path) -> str:
    """A commit that is not an ancestor of anything else, for force-push tests."""
    _git(repo, "checkout", "-q", "--orphan", "orphan")
    _git(repo, "rm", "-q", "-rf", ".")
    (repo / "other.txt").write_text("other\n")
    _git(repo, "add", "other.txt")
    _git(repo, "commit", "-q", "-m", "orphan")
    sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-q", "main")
    return sha


# --- origin: public, main only ------------------------------------------------


def test_origin_main_passes(tmp_path):
    repo = _repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    result = _run_hook(repo, "origin", [_line("refs/heads/main", sha)])
    assert result.returncode == 0, result.stderr + result.stdout
    assert "public tree" in result.stdout


def test_origin_refuses_private_deploy(tmp_path):
    repo = _repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    result = _run_hook(repo, "origin", [_line("refs/heads/private-deploy", sha)])
    assert result.returncode != 0
    assert "only main is published" in result.stderr


def test_origin_refuses_any_other_ref(tmp_path):
    repo = _repo(tmp_path)
    _git(repo, "switch", "-q", "-c", "feature")
    sha = _git(repo, "rev-parse", "HEAD")
    result = _run_hook(repo, "origin", [_line("refs/heads/feature", sha)])
    assert result.returncode != 0
    assert "only main is published" in result.stderr


def test_origin_refuses_artifact_history(tmp_path):
    repo = _repo(tmp_path)
    art = repo / "artifacts" / "serve"
    art.mkdir(parents=True)
    (art / "svm_bundle.joblib").write_bytes(b"fixture")
    _git(repo, "add", "-f", "artifacts/serve/svm_bundle.joblib")
    _git(repo, "commit", "-q", "-m", "artifact")
    sha = _git(repo, "rev-parse", "HEAD")
    result = _run_hook(repo, "origin", [_line("refs/heads/main", sha)])
    assert result.returncode != 0
    assert "artifacts/serve" in result.stderr


def test_origin_refuses_a_force_push(tmp_path):
    repo = _repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    orphan = _orphan_sha(repo)
    result = _run_hook(repo, "origin", [_line("refs/heads/main", orphan, remote_sha=sha)])
    assert result.returncode != 0
    assert "fast-forward" in result.stderr


@pytest.mark.parametrize(
    "path",
    ["data/local/real_alerts.csv", "reports/results/real_alerts.json"],
)
def test_origin_refuses_local_real_alerts(tmp_path, path):
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


# --- private: private-deploy -> main only ------------------------------------


def test_private_deploy_to_private_main_passes(tmp_path):
    repo = _repo(tmp_path)
    _add_private_remote(repo)
    sha = _commit_artifacts(repo)
    result = _run_hook(
        repo,
        "private",
        [_line("refs/heads/private-deploy", sha, "refs/heads/main")],
        url=PRIVATE_URL,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "private tree" in result.stdout


def test_private_refuses_main(tmp_path):
    repo = _repo(tmp_path)
    _add_private_remote(repo)
    sha = _git(repo, "rev-parse", "HEAD")
    result = _run_hook(
        repo, "private", [_line("refs/heads/main", sha, "refs/heads/main")], url=PRIVATE_URL
    )
    assert result.returncode != 0
    assert "only private-deploy is published there" in result.stderr


def test_private_refuses_a_different_remote_ref(tmp_path):
    repo = _repo(tmp_path)
    _add_private_remote(repo)
    sha = _commit_artifacts(repo)
    result = _run_hook(
        repo,
        "private",
        [_line("refs/heads/private-deploy", sha, "refs/heads/private-deploy")],
        url=PRIVATE_URL,
    )
    assert result.returncode != 0
    assert "private main" in result.stderr


def test_private_refuses_a_mismatched_url(tmp_path):
    repo = _repo(tmp_path)
    _add_private_remote(repo)
    sha = _commit_artifacts(repo)
    result = _run_hook(
        repo,
        "private",
        [_line("refs/heads/private-deploy", sha, "refs/heads/main")],
        url="ssh://evil.invalid/private.git",
    )
    assert result.returncode != 0
    assert "does not match the configured" in result.stderr


def test_private_refuses_when_no_url_is_configured(tmp_path):
    repo = _repo(tmp_path)
    sha = _commit_artifacts(repo)
    result = _run_hook(
        repo,
        "private",
        [_line("refs/heads/private-deploy", sha, "refs/heads/main")],
        url=PRIVATE_URL,
    )
    assert result.returncode != 0
    assert "no configured URL" in result.stderr


def test_private_refuses_a_force_push(tmp_path):
    repo = _repo(tmp_path)
    _add_private_remote(repo)
    sha = _commit_artifacts(repo)
    orphan = _orphan_sha(repo)
    result = _run_hook(
        repo,
        "private",
        [_line("refs/heads/private-deploy", orphan, "refs/heads/main", remote_sha=sha)],
        url=PRIVATE_URL,
    )
    assert result.returncode != 0
    assert "fast-forward" in result.stderr


# --- anything else ------------------------------------------------------------


@pytest.mark.parametrize("remote", ["upstream", "backup", "Private", ""])
def test_unknown_remote_is_refused(tmp_path, remote):
    repo = _repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    result = _run_hook(repo, remote, [_line("refs/heads/main", sha)])
    assert result.returncode != 0
    assert "unknown remote" in result.stderr


@pytest.mark.parametrize("remote", ["origin", "private"])
def test_deletions_are_ignored(tmp_path, remote):
    repo = _repo(tmp_path)
    if remote == "private":
        _add_private_remote(repo)
    result = _run_hook(repo, remote, [_line("(delete)", ZERO)], url=PRIVATE_URL)
    assert result.returncode == 0
