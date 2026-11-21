"""Regression: app/app.py must import from the repo root without PYTHONPATH.

Streamlit Community Cloud runs ``streamlit run app/app.py`` from the repo root
with only requirements.txt installed, so ``src/`` is not on sys.path. The app
must bootstrap it from ``__file__``; an editable install is not available.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "serve" / "svm_bundle.joblib"

SCRIPT = """
import importlib.util
from pathlib import Path

app_path = Path("app/app.py").resolve()
spec = importlib.util.spec_from_file_location("smishlens_app", app_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from smishlens.demo_examples import EXAMPLES
from smishlens.serve import BAND_HAM, BAND_REVIEW, BAND_SCAM, load_bundle, predict_message

bundle = load_bundle("artifacts/serve")
result = predict_message(bundle, EXAMPLES[0]["text"])
assert result.band in {BAND_HAM, BAND_REVIEW, BAND_SCAM}
assert "Triage aid" in module.NOTICE
print("ok")
"""


def test_app_imports_from_repo_root_without_pythonpath():
    if not BUNDLE.exists():
        pytest.skip("serving bundle is private; nothing to import against")
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
