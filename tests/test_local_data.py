"""Tests for the local-only data guard and the text-free split manifest."""

from __future__ import annotations

import csv

import pytest

from smishlens import local_data


def _write_split(directory, name: str, spam: int, ham: int) -> None:
    """Write a tiny split with the ``y`` column and no message text."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["y", "source"])
        for _ in range(spam):
            writer.writerow(["1", "test"])
        for _ in range(ham):
            writer.writerow(["0", "test"])


@pytest.fixture()
def splits(tmp_path):
    for name, spam, ham in (("train", 3, 5), ("cal", 1, 2), ("test", 2, 2)):
        _write_split(tmp_path, name, spam, ham)
    return tmp_path


def test_counts_reads_only_the_label_column(splits):
    assert local_data.counts(splits / "train.csv") == (8, 3, 5)


def test_manifest_round_trips(splits):
    local_data.write_manifest(splits)
    assert local_data.verify_manifest(splits) == []


def test_manifest_lists_counts_and_digests_only(splits):
    target = local_data.write_manifest(splits)
    body = target.read_text().splitlines()[1:]
    assert len(body) == 3
    expected = {"train.csv": (8, 3, 5), "cal.csv": (3, 1, 2), "test.csv": (4, 2, 2)}
    for line in body:
        digest, name, rows, spam, ham = line.split()
        assert len(digest) == 64
        assert (int(rows), int(spam), int(ham)) == expected[name]


def test_manifest_detects_a_changed_split(splits):
    local_data.write_manifest(splits)
    _write_split(splits, "train", 4, 4)
    problems = local_data.verify_manifest(splits)
    assert any("hash mismatch" in problem for problem in problems)


def test_manifest_reports_a_missing_split(splits):
    local_data.write_manifest(splits)
    (splits / "cal.csv").unlink()
    problems = local_data.verify_manifest(splits)
    assert any("missing split" in problem for problem in problems)


def test_require_uci_exits_with_one_instruction(tmp_path):
    with pytest.raises(SystemExit) as caught:
        local_data.require_uci(tmp_path)
    message = str(caught.value)
    assert "make setup" in message
    assert "train.csv" in message


def test_require_gold_exits_with_one_instruction(tmp_path):
    with pytest.raises(SystemExit) as caught:
        local_data.require_gold(tmp_path)
    assert "local-only" in str(caught.value)


def test_require_file_exits_on_a_missing_path(tmp_path):
    with pytest.raises(SystemExit):
        local_data.require_file(tmp_path / "nope.csv")


def test_require_uci_passes_when_the_splits_exist(splits):
    assert local_data.require_uci(splits) is None
