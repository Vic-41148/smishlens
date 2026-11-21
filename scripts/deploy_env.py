"""Verify the deployment environment and write ``reports/results/deploy_env.json``.

Builds a throwaway virtualenv from ``requirements.txt`` only, moves the pool
CSVs aside (and restores them byte-identically), then checks that the app import
path works, the bundle loads, one fixed synthetic message scores the same under
the clean interpreter and the project ``.venv``, and the Streamlit AppTest
passes. Optionally builds and smokes the Docker image.

No message text is read, printed or written. The synthetic probe is a fixed
invented string; only its band, conformal set and calibrated score are recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "results" / "deploy_env.json"
CSV_DIR = ROOT / "data" / "processed" / "modern"
CSVS = ("train.csv", "cal.csv", "test.csv")
ARTIFACT_DIR = ROOT / "artifacts" / "serve"
SYNTHETIC = "Your parcel is held; pay the fee at example.invalid"
IMAGE = "smishlens-deploy:verify"
CONTAINER = "smishlens-deploy-verify"
HEALTH_URL = "http://127.0.0.1:8599/_stcore/health"
TOLERANCE = 1e-9


def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, **kwargs)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _direct_requirements() -> list[str]:
    names = []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        names.append(line.split("==", 1)[0].strip())
    return names


def _pick_python() -> str:
    for candidate in (os.environ.get("DEPLOY_PYTHON"), "python3.12", "python3"):
        if candidate and shutil.which(candidate):
            return candidate
    return sys.executable


def _make_venv(venv: Path, python: str) -> None:
    _run([python, "-m", "venv", "--clear", str(venv)], check=True)
    _run([str(venv / "bin" / "python"), "-m", "pip", "install", "--upgrade", "pip"], check=True)


def _install(venv: Path) -> None:
    result = _run(
        [
            str(venv / "bin" / "python"),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(ROOT / "requirements.txt"),
        ]
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])


def _python_version(python: str) -> str:
    return _run([python, "-c", "import platform; print(platform.python_version())"], check=True).stdout.strip()


def _versions(venv: Path, names: list[str]) -> dict:
    code = (
        "import json\n"
        "from importlib.metadata import version\n"
        f"names = {names!r}\n"
        "print(json.dumps({n: version(n) for n in names}))\n"
    )
    result = _run([str(venv / "bin" / "python"), "-c", code], check=True)
    return json.loads(result.stdout)


def _probe_script() -> str:
    """Import the app path, load the bundle, score once, then run AppTest."""
    return f"""
import json, os, sys
from pathlib import Path
root = Path({str(ROOT)!r})
sys.path.insert(0, str(root / "src"))
os.environ["SMISHLENS_SERVE_ARTIFACT"] = str(root / "artifacts" / "serve")
os.environ.pop("SMISHLENS_ENABLE_MMBERT", None)

from smishlens.serve import load_bundle, predict_message
import smishlens.demo_examples  # imported by app/app.py
bundle = load_bundle(root / "artifacts" / "serve")
result = predict_message(bundle, {SYNTHETIC!r})
compile((root / "app" / "app.py").read_text(), "app/app.py", "exec")

from streamlit.testing.v1 import AppTest
at = AppTest.from_file(str(root / "app" / "app.py"), default_timeout=60)
at.run()
notice = any("Triage aid" in w.value for w in at.warning)
at.text_area[0].set_value({SYNTHETIC!r})
at.button[0].click().run()
scored = any("Conformal set" in m.label for m in at.metric)
print("RESULT " + json.dumps({{
    "app_import_ok": True,
    "bundle_loaded": True,
    "synthetic_band": result.band,
    "synthetic_set": list(result.conformal_set),
    "synthetic_p_spam": result.p_spam,
    "apptest_ok": bool(not at.exception and notice and scored),
    "apptest_notice": bool(notice),
    "apptest_scored": bool(scored),
}}))
"""


def _score_script() -> str:
    """Load the bundle and score the fixed synthetic message. Nothing else."""
    return f"""
import json, os, sys
from pathlib import Path
root = Path({str(ROOT)!r})
sys.path.insert(0, str(root / "src"))
os.environ["SMISHLENS_SERVE_ARTIFACT"] = str(root / "artifacts" / "serve")
os.environ.pop("SMISHLENS_ENABLE_MMBERT", None)
from smishlens.serve import load_bundle, predict_message
result = predict_message(load_bundle(root / "artifacts" / "serve"), {SYNTHETIC!r})
print("RESULT " + json.dumps({{"band": result.band, "p_spam": result.p_spam}}))
"""


def _parse_result(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise RuntimeError("probe produced no RESULT line")


def _artifact() -> dict:
    path = ARTIFACT_DIR / "svm_bundle.joblib"
    digest = _sha256(path)
    manifest = {}
    for line in (ARTIFACT_DIR / "MANIFEST.sha256").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            manifest[parts[1]] = parts[0]
    return {
        "artifact_sha256": digest,
        "manifest_sha256": manifest.get("svm_bundle.joblib"),
        "artifact_matches_manifest": manifest.get("svm_bundle.joblib") == digest,
    }


def _dockerfile_base() -> str:
    for line in (ROOT / "Dockerfile").read_text().splitlines():
        if line.strip().upper().startswith("FROM "):
            return line.split()[1]
    return "unknown"


def _docker() -> dict:
    if not shutil.which("docker"):
        return {"installed": False, "tested": False, "note": "docker not installed"}
    if _run(["docker", "info"]).returncode:
        return {"installed": True, "tested": False, "note": "docker daemon unavailable"}

    record: dict = {"installed": True, "tested": False, "base_image": _dockerfile_base()}
    try:
        build = _run(["docker", "build", "-t", IMAGE, str(ROOT)])
        if build.returncode:
            record["error"] = (build.stderr or build.stdout)[-1500:]
            return record

        _run(["docker", "rm", "-f", CONTAINER])
        run = _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                CONTAINER,
                "-p",
                "127.0.0.1:8599:8501",
                IMAGE,
            ]
        )
        if run.returncode:
            record["error"] = (run.stderr or run.stdout)[-1500:]
            return record

        health = False
        for _ in range(60):
            try:
                with urllib.request.urlopen(HEALTH_URL, timeout=2) as response:
                    health = response.status == 200
                    if health:
                        break
            except OSError:  # refused/reset while the server boots
                time.sleep(1)

        score = _run(
            [
                "docker",
                "exec",
                CONTAINER,
                "python",
                "-c",
                "import sys; sys.path.insert(0, '/app/src'); "
                "from smishlens.serve import load_bundle, predict_message; "
                "b = load_bundle('/app/artifacts/serve'); "
                f"print(predict_message(b, {SYNTHETIC!r}).band)",
            ]
        )
        no_data = _run(
            [
                "docker",
                "run",
                "--rm",
                IMAGE,
                "python",
                "-c",
                "import pathlib, sys; "
                "root = pathlib.Path('/app'); "
                "bad = [str(p) for p in root.rglob('*.csv')]; "
                "bad += [str(root / n) for n in ('data', 'models', 'checkpoints', 'reports') "
                "if (root / n).exists()]; "
                "bad += [str(p) for p in root.rglob('*.joblib') "
                "if 'artifacts/serve' not in str(p)]; "
                "print('FOUND ' + ' '.join(bad)); sys.exit(1 if bad else 0)",
            ]
        )
        size = _run(["docker", "image", "inspect", "--format", "{{.Size}}", IMAGE])
        size_bytes = int(size.stdout.strip()) if size.returncode == 0 else None
        record.update(
            {
                "tested": True,
                "health_ok": health,
                "synthetic_band": score.stdout.strip() if score.returncode == 0 else None,
                "no_pool_csvs": no_data.returncode == 0,
                "image_size_mb": round(size_bytes / 1_000_000, 1) if size_bytes else None,
            }
        )
        if score.returncode:
            record["score_error"] = (score.stderr or score.stdout)[-1500:]
        if no_data.returncode:
            record["no_pool_csvs_detail"] = (no_data.stdout or "").strip()[-500:]
    except Exception as exc:  # noqa: BLE001 - recorded, cleanup still runs
        record.setdefault("tested", False)
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _run(["docker", "rm", "-f", CONTAINER])
        record["container_removed"] = _run(["docker", "ps", "-aq", "-f", f"name={CONTAINER}"]).stdout.strip() == ""
        _run(["docker", "rmi", "-f", IMAGE])
        record["image_removed"] = _run(["docker", "images", "-q", IMAGE]).stdout.strip() == ""
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", default="/tmp/opencode/smishlens-deploy-venv")
    parser.add_argument("--reference-venv", default=str(ROOT / ".venv"))
    parser.add_argument("--no-docker", action="store_true")
    args = parser.parse_args(argv)

    venv = Path(args.venv)
    reference = Path(args.reference_venv)
    python = _pick_python()
    names = _direct_requirements()
    hold = Path(tempfile.mkdtemp(prefix="smishlens-csv-", dir="/tmp/opencode"))
    moved: list[tuple[Path, Path]] = []
    before: dict[str, str] = {}
    record: dict = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "requirements": "requirements.txt",
        "python_requested": python,
        "host_python": platform.python_version(),
        "csvs_absent": False,
        "csv_restore_verified": False,
        "error": None,
    }
    try:
        _make_venv(venv, python)
        _install(venv)
        record["packages"] = _versions(venv, names)
        record["python_version"] = _python_version(str(venv / "bin" / "python"))

        before = {name: _sha256(CSV_DIR / name) for name in CSVS if (CSV_DIR / name).exists()}
        for name in CSVS:
            source = CSV_DIR / name
            if source.exists():
                target = hold / name
                shutil.move(str(source), str(target))
                moved.append((source, target))
        record["csvs_absent"] = not any((CSV_DIR / name).exists() for name in CSVS)
        record["csvs_moved"] = len(moved)

        probe = _run([str(venv / "bin" / "python"), "-c", _probe_script()])
        if probe.returncode:
            record["probe_error"] = (probe.stderr or probe.stdout)[-2000:]
            raise RuntimeError("clean-venv probe failed")
        record.update(_parse_result(probe.stdout))
        record.update(_artifact())

        clean_score = record["synthetic_p_spam"]
        cross: dict = {"reference_venv": str(reference), "tolerance": TOLERANCE}
        if (reference / "bin" / "python").exists():
            cross["reference_python"] = _python_version(str(reference / "bin" / "python"))
            ref = _run([str(reference / "bin" / "python"), "-c", _score_script()])
            if ref.returncode:
                cross["reference_error"] = (ref.stderr or ref.stdout)[-1500:]
                cross["match"] = False
            else:
                ref_score = _parse_result(ref.stdout)
                cross["reference_p_spam"] = ref_score["p_spam"]
                cross["reference_band"] = ref_score["band"]
                cross["max_abs_diff"] = abs(ref_score["p_spam"] - clean_score)
                cross["match"] = bool(
                    cross["max_abs_diff"] <= TOLERANCE
                    and ref_score["band"] == record["synthetic_band"]
                )
        else:
            cross["match"] = False
            cross["reference_error"] = "reference venv missing"
        record["cross_version"] = cross
    except Exception as exc:  # noqa: BLE001 - recorded, then surfaced as exit 1
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for source, target in moved:
            if target.exists():
                shutil.move(str(target), str(source))
        shutil.rmtree(hold, ignore_errors=True)
        after = {name: _sha256(CSV_DIR / name) for name in CSVS if (CSV_DIR / name).exists()}
        record["csv_restore_verified"] = bool(before) and before == after
        if not record["csv_restore_verified"]:
            record["error"] = record["error"] or "CSV restore check failed"

    if record["error"] is None and not args.no_docker:
        record["docker"] = _docker()
    else:
        record["docker"] = {
            "installed": shutil.which("docker") is not None,
            "tested": False,
            "skipped": True,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")

    failures = [
        name
        for name, ok in (
            ("csv_restore", record["csv_restore_verified"]),
            ("app_import", record.get("app_import_ok")),
            ("bundle", record.get("bundle_loaded")),
            ("apptest", record.get("apptest_ok")),
            ("artifact_manifest", record.get("artifact_matches_manifest")),
            ("cross_version", record.get("cross_version", {}).get("match")),
        )
        if ok is not True
    ]
    if record["error"]:
        failures.insert(0, "error")
    if failures:
        print("FAILED checks: " + ", ".join(failures))
        return 1
    print("all deployment-environment checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
