"""Loaders, conflict handling, splits and the dataset fingerprint."""

import pandas as pd

from smishlens.data import (
    SCHEMA,
    clean_dedup,
    dataset_hash,
    finalize,
    load_generic,
    load_uci,
    split_pool,
)


def test_load_uci(tmp_path):
    path = tmp_path / "sms.tsv"
    path.write_text("ham\tHello there\nspam\tWin cash now\nham\tAnother ham\n")
    df = load_uci(path)
    assert list(df.columns) == SCHEMA
    assert df["y"].tolist() == [0, 1, 0]
    assert df["y3"].tolist() == ["ham", "spam", "ham"]
    assert (df["source"] == "uci").all()


def test_finalize_sets_url_flag():
    df = finalize(pd.DataFrame({"text": ["see http://x.test now", "nothing here"]}), "demo")
    assert df["has_url"].tolist() == [True, False]
    assert list(df.columns) == SCHEMA


def test_clean_dedup_drops_conflicts_and_duplicates():
    df = pd.DataFrame(
        {
            "text": ["a", "a", "b", "b"],
            "y": [0, 1, 0, 0],
            "y3": ["ham", "spam", "ham", "ham"],
            "source": ["x", "y", "x", "y"],
        }
    )
    out = clean_dedup(df, priority=["x", "y"])
    assert out["text"].tolist() == ["b"]
    assert out["source"].tolist() == ["x"]


def test_split_pool_sizes_and_classes(sms_df):
    train, cal, test = split_pool(sms_df)
    assert len(train) + len(cal) + len(test) == len(sms_df)
    assert len(train) > len(test) > len(cal) > 0
    assert train["y"].nunique() == 2


def test_dataset_hash_is_order_independent(sms_df):
    assert dataset_hash(sms_df) == dataset_hash(sms_df.sample(frac=1, random_state=1))


def test_load_generic_label_map():
    raw = pd.DataFrame({"body": ["hi", "win cash"], "tag": ["Ham", "Spam"]})
    out = load_generic(raw, "demo", "body", "tag", {"ham": "ham", "spam": "spam"}, lang="en")
    assert out["y"].tolist() == [0, 1]
    assert (out["source"] == "demo").all()
    assert (out["lang"] == "en").all()
