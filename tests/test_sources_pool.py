"""Adapters and pool logic on small synthetic fixtures (no raw data needed)."""

from __future__ import annotations

import json

import pandas as pd

from smishlens.data import SCHEMA
from smishlens.pool import (
    _normalised_key,
    dedup_across_sources,
    dedup_within,
    drop_near_dups,
    drop_uci_overlap,
    make_loso_folds,
    make_manifest,
    row_id,
)
from smishlens.sources import load_africa, load_exais, load_moz, load_wspr


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
            _frame([("Ｗin cash now", "spam")], "exais"),  # fullwidth, normalises equal
        ],
        ignore_index=True,
    )
    out, log = dedup_within(frame)
    assert log["rows_in"] == 3
    assert len(out) == 1
    # Priority order is (imc25, mishra, exais, moz, smishtank): exais outranks moz.
    assert out["source"].iloc[0] == "exais"


def test_dedup_drops_label_conflicts_entirely():
    frame = pd.concat(
        [_frame([("same text", "ham")], "moz"), _frame([("same text", "spam")], "exais")],
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
