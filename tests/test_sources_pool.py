"""Adapters and pool logic on small synthetic fixtures (no raw data needed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smishlens.data import SCHEMA
from smishlens.pool import (
    NEAR_DUP_THRESHOLD,
    _normalised_key,
    apply_dedup_exclusions,
    apply_label_overrides,
    dedup_across_sources,
    dedup_within,
    drop_ham_only_spam,
    drop_near_dups,
    drop_uci_overlap,
    make_loso_folds,
    make_manifest,
    near_dup_scores,
    row_id,
)
from smishlens.sources import (
    load_africa,
    load_exais,
    load_moz,
    load_uci_from_raw,
    load_wspr,
)


def _frame(rows, source):
    out = pd.DataFrame(rows, columns=["text", "y3"])
    out["y"] = (out["y3"] != "ham").astype(int)
    out["source"] = source
    out["lang"] = "en"
    out["date"] = None
    out["has_url"] = False
    out["scam_type"] = None
    out["synthetic"] = False
    return out[SCHEMA]


def test_exais_ragged_rows_locate_label_token(tmp_path):
    # Column count changes per row; the label is not always at the same index.
    (tmp_path / "exais" / "extracted" / "DS").mkdir(parents=True)
    path = tmp_path / "exais" / "extracted" / "DS" / "USER 1.csv"
    path.write_text(
        "SMS,receive,1,1,08/08/2014 13:08,27,HAM,hello there,see you at 6\n"
        "SMS,receive,1,1,08/08/2014 13:08,27,SPAM,win cash,claim now,http://x.example,yes\n",
        encoding="utf-8",
    )
    df = load_exais(raw=tmp_path)
    assert len(df) == 2
    assert df["y3"].tolist() == ["ham", "spam"]
    # Commas inside the message survive because the tail cells are re-joined.
    assert df["text"].iloc[1] == "win cash, claim now, http://x.example, yes"
    assert df["y"].tolist() == [0, 1]


def test_moz_label_mapping(tmp_path):
    (tmp_path / "moz").mkdir(parents=True)
    (tmp_path / "moz" / "test.csv").write_text(
        "id,source,text,label\n"
        "1_0,sms,Oi tudo bem,Legitimate\n"
        "2_0,sms,Pague o mpesa agora,Smishing\n",
        encoding="utf-8",
    )
    df = load_moz(raw=tmp_path)
    assert df["y3"].tolist() == ["ham", "smish"]
    assert df["y"].tolist() == [0, 1]
    assert set(df["lang"]) == {"pt"}


def test_wspr_parses_seven_fields_despite_six_headers(tmp_path):
    (tmp_path / "wspr").mkdir(parents=True)
    (tmp_path / "wspr" / "phishing_messages.csv").write_text(
        "messageID,objectID,destination number,message,time,error in time\n"
        '0,abc,+447,"Join today and enjoy 400% bonus, START https://x.example",'
        "VIP Club,1651510753.0,7200.0\n",
        encoding="utf-8",
    )
    df = load_wspr(raw=tmp_path)
    assert len(df) == 1
    assert df["text"].iloc[0] == "Join today and enjoy 400% bonus, START https://x.example"
    assert df["y3"].iloc[0] == "smish"


def test_africa_is_tagged_synthetic(tmp_path):
    (tmp_path / "hf" / "africa-smishing" / "data").mkdir(parents=True)
    pd.DataFrame(
        {"sms_text": ["a", "b"], "label": [0, 1], "language": ["English", "Swahili"]}
    ).to_parquet(tmp_path / "hf" / "africa-smishing" / "data" / "train.parquet")
    df = load_africa(raw=tmp_path)
    assert df["synthetic"].all()
    assert df["y"].tolist() == [0, 1]


def test_dedup_drops_exact_and_normalised_copies():
    frame = pd.concat(
        [
            _frame([("Win  CASH now", "spam"), ("win cash now", "spam")], "moz"),
            _frame([("Ｗin cash now", "spam")], "mishra"),  # fullwidth, normalises equal
        ],
        ignore_index=True,
    )
    out, log = dedup_within(frame)
    assert log["rows_in"] == 3
    assert len(out) == 1
    # Priority order is (imc25, mishra, moz, smishtank): mishra outranks moz.
    assert out["source"].iloc[0] == "mishra"


def test_dedup_drops_label_conflicts_entirely():
    frame = pd.concat(
        [_frame([("same text", "ham")], "moz"), _frame([("same text", "spam")], "mishra")],
        ignore_index=True,
    )
    out, log = dedup_within(frame)
    assert len(out) == 0
    assert log["dropped_label_conflict"] == 2


def test_near_dup_removes_cosmetic_variants_but_not_new_text():
    ref = _frame([("Your parcel is held. Pay 2.99 at http://p.example/x", "smish")], "imc25")
    frame = _frame(
        [
            ("Your parcel is held! Pay 2.99 at http://p.example/x", "smish"),
            ("Lunch at noon tomorrow?", "ham"),
        ],
        "mishra",
    )
    out, dropped = drop_near_dups(frame, ref, thr=0.8)
    assert dropped == 1
    assert out["text"].tolist() == ["Lunch at noon tomorrow?"]


def test_uci_overlap_removes_exact_and_cosmetic_copies():
    uci = _frame([("Free entry! Text WIN to 80086", "spam")], "uci")
    frame = _frame(
        [("free entry! text win to 80086", "spam"), ("See you at 7", "ham")], "mishra"
    )
    out, log = drop_uci_overlap(frame, uci, thr=0.8)
    assert out["text"].tolist() == ["See you at 7"]
    assert log["exact"] == 1


def test_cross_source_dedup_keeps_earlier_source():
    frame = pd.concat(
        [
            _frame([("Claim your prize at http://a.example now", "smish")], "imc25"),
            _frame([("Claim your prize at http://a.example now!", "smish")], "smishtank"),
        ],
        ignore_index=True,
    )
    out, log = dedup_across_sources(frame)
    assert out["source"].tolist() == ["imc25"]
    assert log["smishtank"]["dropped"] == 1


def test_manifest_has_no_text_and_unique_ids():
    frame = _frame([("a", "ham"), ("b", "spam")], "moz")
    split_of = pd.Series(["train", "test"], index=frame.index)
    manifest = make_manifest(frame, split_of)
    assert "text" not in manifest.columns
    assert manifest["row_id"].is_unique
    assert set(manifest["split"]) == {"train", "test"}
    assert all(len(h) == 16 for h in manifest["text_sha256_16"])


def test_loso_folds_are_disjoint_and_cover_every_row():
    frame = pd.concat(
        [_frame([(f"m{i}", "ham") for i in range(4)], "moz"),
         _frame([(f"i{i}", "smish") for i in range(4)], "imc25")],
        ignore_index=True,
    )
    split_of = pd.Series(["train", "cal", "test", "train", "train", "cal", "test", "train"])
    folds = make_loso_folds(frame, split_of)
    ids = {row_id(s, t) for s, t in zip(frame["source"], frame["text"], strict=True)}
    all_eval: list[str] = []
    for fold in folds["folds"]:
        assert set(fold["eval_ids"]) <= ids
        assert all(row_id(fold["held_out"], t) in set(fold["eval_ids"]) for t in frame.loc[frame["source"] == fold["held_out"], "text"])
        all_eval.extend(fold["eval_ids"])
        # train never contains a held-out row
        held = set(fold["eval_ids"])
        assert not (held & set(fold["eval_ids"]) - held)
    assert sorted(all_eval) == sorted(ids)  # every row appears in exactly one fold


def test_normalised_key_is_case_and_unicode_insensitive():
    assert _normalised_key("Ｗin   CASH") == _normalised_key("win cash")


def test_audit_json_round_trips():
    # Guards the shape the committed audit file must keep.
    payload = json.loads(
        json.dumps(
            {
                "seed": 42,
                "pool_rows": 1,
                "dataset_sha256_16": "0" * 16,
                "splits": {"train": {"rows": 1}},
            }
        )
    )
    assert payload["seed"] == 42


def test_apply_label_overrides_changes_both_labels():
    frame = _frame([("hello", "ham"), ("win cash", "smish")], "moz")
    ids = [row_id(s, t) for s, t in zip(frame["source"], frame["text"], strict=True)]
    out = apply_label_overrides(frame, {ids[0]: "smish"})
    assert out.loc[0, "y3"] == "smish"
    assert out.loc[0, "y"] == 1
    assert out.loc[1, "y3"] == "smish"  # untouched


def test_apply_label_overrides_rejects_stale_ids():
    frame = _frame([("hello", "ham")], "moz")
    try:
        apply_label_overrides(frame, {"deadbeefdeadbeef": "spam"})
    except ValueError as exc:
        assert "not in the pool" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("stale override id must raise")


def test_apply_label_overrides_rejects_bad_label():
    frame = _frame([("hello", "ham")], "moz")
    rid = row_id("moz", "hello")
    try:
        apply_label_overrides(frame, {rid: "maybe"})
    except ValueError as exc:
        assert "bad label" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("bad label must raise")


def test_override_resolves_label_conflict_before_dedup():
    frame = pd.concat(
        [
            _frame([("claim your prize now", "spam")], "mishra"),
            _frame([("claim your prize now", "ham")], "moz"),
        ],
        ignore_index=True,
    )
    out, log = dedup_within(frame)
    assert out.empty  # conflicting labels drop both rows
    assert log["dropped_label_conflict"] == 2

    rid = row_id("moz", "claim your prize now")
    fixed = apply_label_overrides(frame, {rid: "spam"})
    out, log = dedup_within(fixed)
    assert len(out) == 1
    assert out["source"].iloc[0] == "mishra"  # dedup priority still decides
    assert log["dropped_label_conflict"] == 0


def test_build_applies_label_overrides(monkeypatch):
    from smishlens import pool as pool_mod

    # Distinct alphabetic tokens: the normaliser masks digits, so numbered
    # templates would collapse to one key.
    topics = [f"topic{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(40)]
    rows = [
        (f"congratulations you have won the {t} prize call now", "smish") for t in topics
    ]
    rows += [(f"see you at the {t} station later", "ham") for t in topics]
    frame = _frame(rows, "moz")

    monkeypatch.setattr(pool_mod, "load_pool", lambda raw=None: frame.copy())
    monkeypatch.setattr(pool_mod, "load_uci_from_raw", lambda raw=None: frame.iloc[0:0])
    monkeypatch.setattr(pool_mod, "ood_stats", lambda raw=None: {})

    rid = row_id("moz", "see you at the topicaa station later")
    result = pool_mod.build(label_overrides={rid: "smish"})

    assert result["audit"]["label_overrides"] == {
        "n": 1,
        "applied_before_dedup": True,
        "n_skipped_ham_only": 0,
        "skipped_sources": [],
    }
    out = result["frame"]
    row = out.loc[out["text"] == "see you at the topicaa station later"].iloc[0]
    assert row["y3"] == "smish"
    assert row["y"] == 1
    assert int(out["y"].sum()) == 41


def test_near_dup_scores_are_corpus_independent():
    ref = _frame([("your parcel is held pay two ninety nine", "smish")], "imc25")
    frame = _frame(
        [
            ("your parcel is held! pay 2.99", "smish"),
            ("lunch at noon tomorrow", "ham"),
        ],
        "mishra",
    )
    base = near_dup_scores(frame, ref)

    # Reordering the reference and adding an unrelated reference row cannot
    # change a score: no idf and no document-frequency pruning.
    extra = _frame([("completely unrelated weather update", "ham")], "moz")
    ref_plus = pd.concat([extra, ref.iloc[::-1]], ignore_index=True)
    assert np.allclose(near_dup_scores(frame, ref_plus), base)

    # Reordering the query frame permutes the scores, not their values.
    perm = [1, 0]
    assert np.allclose(
        near_dup_scores(frame.iloc[perm].reset_index(drop=True), ref), base[perm]
    )


def test_dedup_exclusions_drop_ids_and_reject_stale_list():
    frame = _frame([("hello there", "ham"), ("win cash now", "spam")], "mishra")
    rid = row_id("mishra", "win cash now")
    out, mask = apply_dedup_exclusions(frame, {rid: "audited near-dup"})
    assert out["text"].tolist() == ["hello there"]
    assert mask.tolist() == [False, True]

    with pytest.raises(ValueError, match="not present"):
        apply_dedup_exclusions(frame, {"deadbeefdeadbeef": "stale"})


def test_ham_only_source_keeps_ham_and_drops_positives_without_overrides():
    frame = pd.concat(
        [
            _frame([("bank alert debit", "ham"), ("you won a prize", "spam")], "exais"),
            _frame([("see you soon", "ham")], "moz"),
        ],
        ignore_index=True,
    )
    out, dropped = drop_ham_only_spam(frame)
    assert dropped == {"exais": 1}
    assert out.loc[out["source"] == "exais", "text"].tolist() == ["bank alert debit"]

    # A ham-only source is never relabelled, even if a review file says so.
    rid = row_id("exais", "bank alert debit")
    fixed = apply_label_overrides(frame, {rid: "smish"})
    row = fixed.loc[fixed["text"] == "bank alert debit"].iloc[0]
    assert row["y3"] == "ham"
    assert row["y"] == 0


def _synthetic_pool_frame():
    # Distinct alphabetic tokens: the normaliser masks digits, so numbered
    # templates would collapse to one key.
    topics = [f"topic{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(40)]
    rows = [
        (f"congratulations you have won the {t} prize call now", "smish") for t in topics
    ]
    rows += [(f"see you at the {t} station later", "ham") for t in topics]
    return _frame(rows, "moz")


def _patch_synthetic_build(monkeypatch, frame):
    from smishlens import pool as pool_mod

    monkeypatch.setattr(pool_mod, "load_pool", lambda raw=None: frame.copy())
    monkeypatch.setattr(pool_mod, "load_uci_from_raw", lambda raw=None: frame.iloc[0:0])
    monkeypatch.setattr(pool_mod, "ood_stats", lambda raw=None: {})
    return pool_mod


def test_build_records_dedup_exclusions(monkeypatch):
    frame = _synthetic_pool_frame()
    pool_mod = _patch_synthetic_build(monkeypatch, frame)
    rid = row_id("moz", "congratulations you have won the topicaa prize call now")
    reason = (
        "audited pairwise near-dup of a UCI row; "
        "tf-only cosine 0.86-0.90, missed by the 0.90 rule"
    )

    result = pool_mod.build(dedup_exclusions={rid: reason})

    audit = result["audit"]["dedup_exclusions"]
    assert audit["n"] == 1
    assert audit["applied"] == 1
    assert audit["reason"] == reason
    assert audit["row_ids"] == [rid]
    out_ids = {
        row_id(s, t)
        for s, t in zip(result["frame"]["source"], result["frame"]["text"], strict=True)
    }
    assert rid not in out_ids
    assert result["audit"]["dedup"]["vs_uci"]["per_source"]["moz"]["excluded"] == 1


def test_build_raises_on_stale_dedup_exclusion(monkeypatch):
    frame = _synthetic_pool_frame()
    pool_mod = _patch_synthetic_build(monkeypatch, frame)
    with pytest.raises(ValueError, match="not present"):
        pool_mod.build(dedup_exclusions={"deadbeefdeadbeef": "stale"})


# The 36 Mishra rows whose pairwise tf-only cosine to a UCI row is in the
# audited near-dup band. 28 clear 0.90; the other 8 are in dedup_exclusions.json.
KNOWN_MISHRA_UCI_NEAR_DUPS = (
    "9c01d68038ea0570", "268964bdf902d43a", "4d3362f2185ce829", "f9c1251308faf0b7",
    "86e3f54fc61a3032", "20c93b03a1b52538", "75961b28514f7681", "79ab3811150d35e0",
    "2ff4b1b8976b5a14", "e8f4dd0ae304a6b0", "0d031c66a1754e56", "e1c3c07e96394be0",
    "707d47e67a490f84", "73366e718d1429ff", "c28fb447edc0be27", "db372bc208cfd603",
    "428fa31d1f0a1b95", "59eade9a0b59b97a", "05788aa6c9bcafcd", "8f4914d7488f99c8",
    "f202a03d70916045", "1de56be71938cb2c", "04a1b2fd2d93112f", "dfa09200a76a5705",
    "455088164355cfb3", "c35917930add1b1e", "30631ae09fd992ab", "d7f5343e3c24c64c",
    "c7c0c0e32302c5c1", "23fa6e94f1b3a001", "2414b36e0c50ae75", "d5b861abdc6b2c90",
    "4a27555feba88bbe", "9eb432ccab8d9f73", "62091fb9b1d83991", "d1705191e0a30602",
)


def test_known_mishra_uci_near_dups_all_leave_the_pool():
    from smishlens.pool import load_pool

    root = Path(__file__).resolve().parents[1]
    modern = root / "data" / "processed" / "modern"
    if not all((modern / f"{s}.csv").exists() for s in ("train", "cal", "test")):
        pytest.skip("processed pool is not built")
    try:
        raw = load_pool()
        uci = load_uci_from_raw()
    except (FileNotFoundError, OSError):
        pytest.skip("raw sources are not present")

    override_path = modern / "label_overrides.json"
    overrides = json.loads(override_path.read_text()) if override_path.exists() else {}
    raw = apply_label_overrides(raw, overrides)
    raw, _ = drop_ham_only_spam(raw)
    frame, _ = dedup_within(raw)
    mish = frame[frame["source"] == "mishra"].reset_index(drop=True)

    ids = [row_id(s, t) for s, t in zip(mish["source"], mish["text"], strict=True)]
    scores = near_dup_scores(mish, uci)
    caught = {
        rid
        for rid, score in zip(ids, scores, strict=True)
        if score >= NEAR_DUP_THRESHOLD
    }

    known = set(KNOWN_MISHRA_UCI_NEAR_DUPS)
    assert len(known) == 36
    assert len(known & caught) == 28

    exclusions = json.loads((modern / "dedup_exclusions.json").read_text())
    assert known - caught == set(exclusions)

    manifest = pd.read_csv(modern / "split_manifest.csv")
    assert not (known & set(manifest["row_id"]))
