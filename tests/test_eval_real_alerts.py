"""Real-alert harness: aggregate-only reporting, no text, no refit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_real_alerts.py"


def _module():
    spec = importlib.util.spec_from_file_location("eval_real_alerts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Result:
    def __init__(self, band: str, p_spam: float):
        self.band = band
        self.p_spam = p_spam


def _frame(n_ham: int = 12, n_spam: int = 3) -> pd.DataFrame:
    rows = [{"text": f"invented ham fixture {i}", "label": "ham"} for i in range(n_ham)]
    rows += [{"text": f"invented spam fixture {i}", "label": "spam"} for i in range(n_spam)]
    return pd.DataFrame(rows)


def test_evaluate_reports_aggregates_only():
    mod = _module()
    frame = _frame()

    def predict(text: str):
        return Result(mod.BAND_SCAM, 0.9) if "spam" in text else Result(mod.BAND_HAM, 0.1)

    report = mod.evaluate(frame, predict, 0.5)

    assert report["n"] == 15
    assert report["per_label"] == {"ham": 12, "spam": 3}
    assert report["bands"][mod.BAND_HAM] == 12
    assert report["bands"][mod.BAND_SCAM] == 3
    assert report["fpr_at_frozen_threshold"] == 0.0
    assert report["recall_at_frozen_threshold"] == 1.0
    assert report["mean_calibrated_score"]["ham"] == pytest.approx(0.1)
    assert report["mean_calibrated_score"]["spam"] == pytest.approx(0.9)


def test_evaluate_counts_false_positives_at_the_threshold():
    mod = _module()
    frame = _frame(n_ham=10, n_spam=0)

    def predict(text: str):
        index = int(text.rsplit(" ", 1)[1])
        return Result(mod.BAND_SCAM if index < 2 else mod.BAND_HAM, 0.9 if index < 2 else 0.1)

    report = mod.evaluate(frame, predict, 0.5)
    assert report["fpr_at_frozen_threshold"] == pytest.approx(0.2)
    assert report["recall_at_frozen_threshold"] is None
    assert report["bands_by_label"]["ham"][mod.BAND_SCAM] == 2


def test_evaluate_refuses_a_thin_ham_sample():
    mod = _module()
    with pytest.raises(mod.InsufficientSample):
        mod.evaluate(_frame(n_ham=9, n_spam=3), lambda _: Result(mod.BAND_HAM, 0.1), 0.5)


def test_report_contains_no_message_text():
    mod = _module()
    report = mod.evaluate(_frame(), lambda _: Result(mod.BAND_HAM, 0.1), 0.5)
    blob = json.dumps(report)
    assert "invented ham fixture" not in blob
    assert "invented spam fixture" not in blob
    assert "text" not in report


def test_read_alerts_rejects_unknown_labels(tmp_path):
    mod = _module()
    path = tmp_path / "real_alerts.csv"
    pd.DataFrame({"text": ["x"], "label": ["maybe"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        mod.read_alerts(path)


def test_read_alerts_requires_columns(tmp_path):
    mod = _module()
    path = tmp_path / "real_alerts.csv"
    pd.DataFrame({"body": ["x"], "label": ["ham"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        mod.read_alerts(path)


def test_main_refuses_before_loading_the_bundle(tmp_path, capsys):
    mod = _module()
    path = tmp_path / "real_alerts.csv"
    _frame(n_ham=9, n_spam=3).to_csv(path, index=False)
    out = tmp_path / "real_alerts.json"

    code = mod.main(["--input", str(path), "--out", str(out)])

    assert code == 2
    assert not out.exists()
    captured = capsys.readouterr()
    assert "invented ham fixture" not in captured.out + captured.err
    assert "refusing" in captured.out


def test_main_reports_missing_input(tmp_path, capsys):
    mod = _module()
    code = mod.main(["--input", str(tmp_path / "absent.csv"), "--out", str(tmp_path / "o.json")])
    assert code == 1
    assert "nothing to evaluate" in capsys.readouterr().out


def test_main_rejects_bad_labels_without_a_traceback(tmp_path, capsys):
    mod = _module()
    path = tmp_path / "real_alerts.csv"
    pd.DataFrame({"text": ["x"], "label": ["maybe"]}).to_csv(path, index=False)
    code = mod.main(["--input", str(path), "--out", str(tmp_path / "o.json")])
    assert code == 3
    captured = capsys.readouterr()
    assert "cannot use" in captured.out
    assert "Traceback" not in captured.out + captured.err


def test_main_rejects_missing_columns_without_a_traceback(tmp_path, capsys):
    mod = _module()
    path = tmp_path / "real_alerts.csv"
    pd.DataFrame({"body": ["x"], "label": ["ham"]}).to_csv(path, index=False)
    code = mod.main(["--input", str(path), "--out", str(tmp_path / "o.json")])
    assert code == 3
    assert "missing columns" in capsys.readouterr().out


def test_ignore_files_exclude_the_local_set():
    root = Path(__file__).resolve().parents[1]
    gitignore = (root / ".gitignore").read_text()
    dockerignore = (root / ".dockerignore").read_text()
    for name in ("data/local/", "reports/results/real_alerts.json"):
        assert name in gitignore
    assert "data/local" in dockerignore
    assert "reports/results/real_alerts.json" in dockerignore


def test_summary_says_not_yet_run_without_the_json(tmp_path, monkeypatch):
    from smishlens import results_summary

    monkeypatch.setattr(results_summary, "RESULTS", tmp_path)
    text = "\n".join(results_summary.real_alerts_lines())
    assert "Real alerts" in text
    assert "Not yet run" in text


def test_summary_renders_aggregates_only(tmp_path, monkeypatch):
    from smishlens import results_summary

    payload = {
        "n": 14,
        "per_label": {"ham": 11, "spam": 3},
        "bands": {"likely scam": 3, "looks ordinary": 9, "needs review": 2},
        "bands_by_label": {
            "ham": {"likely scam": 1, "looks ordinary": 9, "needs review": 1},
            "spam": {"likely scam": 2, "looks ordinary": 0, "needs review": 1},
        },
        "fpr_at_frozen_threshold": 0.090909,
        "recall_at_frozen_threshold": 0.666667,
        "mean_calibrated_score": {"all": 0.2, "ham": 0.1, "spam": 0.6},
        "min_ham_required": 10,
        "generated_utc": "2026-11-22T00:00:00+00:00",
        "artifact_sha256": "deadbeef",
    }
    (tmp_path / "real_alerts.json").write_text(json.dumps(payload))
    monkeypatch.setattr(results_summary, "RESULTS", tmp_path)
    text = "\n".join(results_summary.real_alerts_lines())
    assert "0.0909" in text
    assert "0.6667" in text
    assert "invented" not in text


def test_readme_marks_the_local_set_as_evaluation_only():
    from smishlens import docs_gen

    text = docs_gen.readme()
    assert "data/local/real_alerts.csv" in text
    assert "evaluation set" in text
    assert "never" in text

