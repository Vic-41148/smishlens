"""Label-audit parsing, Wilson intervals and the 5% gate."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from smishlens.audit import load_review, parse_label, summarise, wilson


def test_parse_label_blank_means_agree():
    assert parse_label("") is None
    assert parse_label("  ") is None
    assert parse_label(None) is None
    assert parse_label(float("nan")) is None


def test_parse_label_maps_aliases():
    assert parse_label("HAM") == "ham"
    assert parse_label("Smishing") == "smish"
    assert parse_label("legitimate") == "ham"
    with pytest.raises(ValueError):
        parse_label("maybe")


def test_wilson_bounds():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.30
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and 0.65 < lo < 0.75
    assert math.isnan(wilson(0, 0)[0])


def _frame():
    return pd.DataFrame(
        {
            "review_id": ["R1", "R2", "R3", "R4", "R5", "R6"],
            "row_id": [f"r{i}" for i in range(6)],
            "source": ["moz", "moz", "moz", "exais", "exais", "exais"],
            "text": ["a", "b", "c", "d", "e", "f"],
            "y": [0, 1, 0, 1, 1, 0],
            "y3": ["ham", "smish", "ham", "spam", "spam", "ham"],
            "reviewer_label": ["", "spam", "smish", "", "ham", ""],
        }
    )


def test_summarise_counts_binary_and_subtype(tmp_path):
    path = tmp_path / "review.csv"
    _frame().to_csv(path, index=False)
    summary, disagree = summarise(load_review(path))
    moz = summary.set_index("source").loc["moz"]
    assert moz["reviewed"] == 3
    assert moz["disagree"] == 1  # R3 ham -> smish
    assert moz["subtype_only"] == 1  # R2 smish -> spam
    exais = summary.set_index("source").loc["exais"]
    assert exais["reviewed"] == 3
    assert exais["disagree"] == 1  # R5 spam -> ham
    assert len(disagree) == 3


def test_load_review_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("review_id,row_id\nR1,r1\n")
    with pytest.raises(ValueError):
        load_review(path)


# --- gate behaviour of data/apply_audit.py ---------------------------------


def _apply_audit_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "apply_audit.py"
    spec = importlib.util.spec_from_file_location("apply_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _review_file(path, n=100, source="moz", flips=(), subtype_flips=()):
    rows = []
    for i in range(n):
        stored = "smish" if i % 2 else "ham"
        label = ""
        if i in flips:
            label = "ham" if stored != "ham" else "smish"
        elif i in subtype_flips:
            label = "spam" if stored == "smish" else ""
        rows.append(
            {
                "review_id": f"R{i}",
                "row_id": f"r{i}",
                "source": source,
                "text": f"message {i}",
                "y": int(stored != "ham"),
                "y3": stored,
                "reviewer_label": label,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_apply_audit_blocks_above_gate_and_writes_nothing(tmp_path, monkeypatch):
    review = _review_file(tmp_path / "review.csv", flips=set(range(10)))
    out = tmp_path / "out"
    import sys

    module = _apply_audit_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["apply_audit.py", "--review", str(review), "--out", str(out),
         "--log", str(tmp_path / "log.md")],
    )
    assert module.main() == 2
    assert not out.exists() or not any(out.iterdir())
    assert not (tmp_path / "log.md").exists()


def test_apply_audit_report_only_below_gate(tmp_path, monkeypatch):
    review = _review_file(tmp_path / "review.csv", flips={0})
    out = tmp_path / "out"
    import sys

    module = _apply_audit_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["apply_audit.py", "--review", str(review), "--out", str(out),
         "--log", str(tmp_path / "log.md")],
    )
    assert module.main() == 0
    assert not out.exists() or not any(out.iterdir())


def test_apply_audit_missing_review_file(tmp_path, monkeypatch):
    import sys

    module = _apply_audit_module()
    monkeypatch.setattr(
        sys, "argv", ["apply_audit.py", "--review", str(tmp_path / "nope.csv")]
    )
    assert module.main() == 1


def test_apply_audit_applies_binary_and_subtype_corrections(tmp_path, monkeypatch):
    review = _review_file(tmp_path / "review.csv", flips={0}, subtype_flips={1})
    out = tmp_path / "out"
    out.mkdir()
    (out / "data_audit.json").write_text('{"dataset_sha256_16": "oldhash"}')
    log = tmp_path / "log.md"
    log.write_text("# log\n")

    import sys

    module = _apply_audit_module()
    captured = {}

    def fake_build(label_overrides=None):
        captured["overrides"] = label_overrides
        return {
            "audit": {
                "dataset_sha256_16": "newhash",
                "pool_rows": 10,
                "splits": {"train": {"rows": 6}, "cal": {"rows": 1}, "test": {"rows": 3}},
            }
        }

    monkeypatch.setattr(module, "build", fake_build)
    monkeypatch.setattr(module, "write_pool", lambda result, path: captured.setdefault("wrote", path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["apply_audit.py", "--review", str(review), "--out", str(out),
         "--log", str(log), "--apply"],
    )
    assert module.main() == 0
    overrides = captured["overrides"]
    assert overrides == {"r0": "smish", "r1": "spam"}
    text = log.read_text()
    assert "oldhash" in text and "newhash" in text and "superseded" in text


def _multi_review(path, spec):
    """spec: {source: (n, flips, subtype_flips)} with unique review ids."""
    rows = []
    for source, (n, flips, subtype_flips) in spec.items():
        for i in range(n):
            stored = "smish" if i % 2 else "ham"
            label = ""
            if i in flips:
                label = "ham" if stored != "ham" else "smish"
            elif i in subtype_flips:
                label = "spam" if stored == "smish" else ""
            rows.append(
                {
                    "review_id": f"{source}-R{i}",
                    "row_id": f"{source}-r{i}",
                    "source": source,
                    "text": f"{source} message {i}",
                    "y": int(stored != "ham"),
                    "y3": stored,
                    "reviewer_label": label,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run(monkeypatch, module, *argv):
    import sys

    monkeypatch.setattr(sys, "argv", ["apply_audit.py", *argv])
    return module.main()


def test_apply_audit_accept_source_passes(tmp_path, monkeypatch):
    review = _review_file(tmp_path / "review.csv", flips=set(range(10)))
    module = _apply_audit_module()
    code = _run(
        monkeypatch, module,
        "--review", str(review),
        "--out", str(tmp_path / "out"),
        "--log", str(tmp_path / "log.md"),
        "--accept-source", "moz:one template, label ambiguity",
    )
    assert code == 0


def test_apply_audit_unnamed_source_still_blocks(tmp_path, monkeypatch):
    review = _multi_review(
        tmp_path / "review.csv",
        {"moz": (100, set(range(10)), ()), "mishra": (100, set(range(10)), ())},
    )
    module = _apply_audit_module()
    code = _run(
        monkeypatch, module,
        "--review", str(review),
        "--out", str(tmp_path / "out"),
        "--log", str(tmp_path / "log.md"),
        "--accept-source", "moz:accepted",
    )
    assert code == 2


def test_apply_audit_skips_dropped_source(tmp_path, monkeypatch, capsys):
    review = _multi_review(
        tmp_path / "review.csv",
        {"exais": (100, set(range(30)), ()), "moz": (100, set(), ())},
    )
    module = _apply_audit_module()
    code = _run(
        monkeypatch, module,
        "--review", str(review),
        "--out", str(tmp_path / "out"),
        "--log", str(tmp_path / "log.md"),
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "ignoring 100 review row(s) from exais" in out


def test_apply_audit_accepted_labels_kept_and_dropped_overrides_filtered(tmp_path, monkeypatch):
    import json

    review = _multi_review(
        tmp_path / "review.csv",
        {
            "exais": (10, {0, 1}, ()),
            "moz": (100, {0, 1}, ()),
            "mishra": (100, {0}, ()),
        },
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / "data_audit.json").write_text('{"dataset_sha256_16": "oldhash"}')
    module = _apply_audit_module()
    captured = {}

    def fake_build(label_overrides=None):
        captured["overrides"] = label_overrides
        return {
            "audit": {
                "dataset_sha256_16": "newhash",
                "pool_rows": 10,
                "splits": {"train": {"rows": 6}, "cal": {"rows": 1}, "test": {"rows": 3}},
            }
        }

    monkeypatch.setattr(module, "build", fake_build)
    monkeypatch.setattr(module, "write_pool", lambda result, path: None)
    code = _run(
        monkeypatch, module,
        "--review", str(review),
        "--out", str(out),
        "--log", str(tmp_path / "log.md"),
        "--accept-source", "moz:label ambiguity",
        "--apply",
    )
    assert code == 0
    overrides = captured["overrides"]
    assert overrides == {"mishra-r0": "smish"}
    assert not any(k.startswith(("exais-", "moz-")) for k in overrides)
    written = json.loads((out / "label_overrides.json").read_text())
    assert written == {"mishra-r0": "smish"}


def test_parse_accept_rejects_bad_format():
    module = _apply_audit_module()
    with pytest.raises(SystemExit):
        module.parse_accept(["moz"])
