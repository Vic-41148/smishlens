"""Generated slides and demo script: shape, sources, and no message text."""

from __future__ import annotations

import json
import re
from pathlib import Path

from smishlens import docs_gen
from smishlens.demo_examples import EXAMPLES

ROOT = Path(__file__).resolve().parents[1]


def _slide_headings(text: str) -> list[str]:
    return [line for line in text.splitlines() if re.match(r"^## \d+\. ", line)]


def test_slides_have_at_most_ten_numbered_slides():
    headings = _slide_headings(docs_gen.slides())
    assert len(headings) == 10


def test_slides_cover_the_required_topics():
    text = docs_gen.slides().lower()
    for topic in (
        "problem",
        "data and licences",
        "method",
        "in-pool",
        "loso",
        "gold",
        "failure analysis",
        "evasion per source",
        "latency",
        "limitations and deployment",
    ):
        assert topic in text, topic


def test_slides_contain_no_demo_message_text():
    text = docs_gen.slides()
    for row in EXAMPLES:
        assert row["text"] not in text


def test_demo_script_answers_the_likely_questions():
    text = docs_gen.demo_script().lower()
    for question in (
        "why no accuracy",
        "gold",
        "mmbert not hosted",
        "artifact private",
    ):
        assert question in text, question


def test_demo_script_uses_only_labelled_demo_examples():
    text = docs_gen.demo_script()
    bands = json.loads((ROOT / "reports" / "results" / "demo_bands.json").read_text())
    labels = {item["label"] for item in bands["rows"]}
    used = [label for label in labels if label in text]
    assert len(used) >= 5
    for row in EXAMPLES:
        assert row["text"] not in text
    assert "Never paste a real" in text


def test_demo_bands_json_holds_no_message_text():
    bands = json.loads((ROOT / "reports" / "results" / "demo_bands.json").read_text())
    assert bands["n_examples"] == len(EXAMPLES)
    for item in bands["rows"]:
        assert set(item) == {"label", "band", "conformal_set", "p_spam"}
        assert isinstance(item["p_spam"], float)
        assert item["band"] in {"likely scam", "looks ordinary", "needs review"}
