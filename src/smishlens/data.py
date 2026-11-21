"""Loaders into one unified schema, conflict handling, dedup, frozen splits."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from smishlens.normalise import URL

SCHEMA = ["text", "y", "y3", "source", "lang", "date", "has_url", "scam_type", "synthetic"]


def finalize(df: pd.DataFrame, source: str, synthetic: bool = False) -> pd.DataFrame:
    """Every loader ends here: same columns, same order, NaN where a source lacks a field."""
    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["source"] = source
    if "y" not in df and "y3" in df:
        df["y"] = (df["y3"] != "ham").astype(int)
    if "synthetic" not in df:
        df["synthetic"] = bool(synthetic)
    df["has_url"] = df["text"].map(lambda s: bool(URL.search(s)))
    for col in SCHEMA:
        if col not in df:
            df[col] = np.nan
    return df[SCHEMA].reset_index(drop=True)


def load_uci(path: str | Path) -> pd.DataFrame:
    """UCI SMS Spam Collection: tab-separated, unquoted, latin-1-ish."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
        encoding_errors="replace",  # NOT errors=
        quoting=csv.QUOTE_NONE,
    )
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["y"] = (df["label"] == "spam").astype(int)
    df["y3"] = df["label"]
    return finalize(df[["text", "y", "y3"]], "uci")


def load_kaggle_spam_csv(path: str | Path) -> pd.DataFrame:
    """The Kaggle mirror of UCI (latin-1, junk unnamed columns)."""
    df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]].rename(
        columns={"v1": "label", "v2": "text"}
    )
    df["y"] = (df["label"].str.lower() == "spam").astype(int)
    df["y3"] = df["label"].str.lower()
    return finalize(df[["text", "y", "y3"]], "uci")


def load_generic(
    df_or_path: str | Path | pd.DataFrame,
    source: str,
    text_col: str,
    label_col: str,
    label_map: dict[str, str],
    **fixed: object,
) -> pd.DataFrame:
    """Template for every modern source.

    label_map maps a raw label to 'ham' | 'spam' | 'smish'. Column names are
    verified per source in DATA_CARD.md before use.
    """
    df = pd.read_csv(df_or_path) if not isinstance(df_or_path, pd.DataFrame) else df_or_path
    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "y3": df[label_col].astype(str).str.strip().str.lower().map(label_map),
        }
    )
    out = out.dropna(subset=["y3"])
    out["y"] = (out["y3"] != "ham").astype(int)
    for key, value in fixed.items():  # e.g. lang="pt", synthetic=False
        out[key] = value
    return finalize(out, source)


def _key(series: pd.Series) -> pd.Series:
    return series.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def clean_dedup(df: pd.DataFrame, priority: list[str] | None = None) -> pd.DataFrame:
    """Drop label conflicts, then exact-duplicate texts keeping the highest-priority source.

    Conflicts are checked across ALL sources: the same text labelled ham in one
    source and spam in another cannot be learned. Run BEFORE splitting.
    """
    df = df.copy()
    df["_k"] = _key(df["text"])
    conflicts = df.groupby("_k")["y"].nunique()
    df = df[~df["_k"].isin(conflicts[conflicts > 1].index)]
    if priority:
        order = {src: i for i, src in enumerate(priority)}
        df = df.assign(_o=df["source"].map(order).fillna(len(order))).sort_values(
            "_o", kind="stable"
        )
        df = df.drop(columns="_o")
    return df.loc[~df["_k"].duplicated()].drop(columns="_k").reset_index(drop=True)


def split_pool(
    df: pd.DataFrame, seed: int = 42, min_stratum: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """64/16/20 stratified by (source, label).

    Strata smaller than min_stratum go to train: they cannot be split.
    """
    strata = df["source"].astype(str) + "|" + df["y"].astype(str)
    counts = strata.map(strata.value_counts())
    small = counts < min_stratum
    big, tiny, big_strata = df[~small], df[small], strata[~small]
    train, rest = train_test_split(big, test_size=0.36, stratify=big_strata, random_state=seed)
    cal, test = train_test_split(
        rest, test_size=0.5556, stratify=strata.loc[rest.index], random_state=seed
    )
    return (
        pd.concat([train, tiny]).reset_index(drop=True),
        cal.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def dataset_hash(df: pd.DataFrame) -> str:
    """Order-independent fingerprint of (text, label) pairs."""
    digest = hashlib.sha256()
    for text, label in sorted(zip(df["text"], df["y"], strict=True)):
        digest.update(f"{text}\t{label}\n".encode())
    return digest.hexdigest()[:16]
