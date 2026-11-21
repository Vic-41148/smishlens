"""Standard published perturbation families, for evaluating and hardening THIS
detector only (RQ3, RQ6).

Ethics: this module exists to measure your own robustness and to show the
normaliser earns its place. Do not package it as a bypass tool and do not craft
new real-world phishing content.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from smishlens.normalise import URL

ZW = "\u200b"
HOMO = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440", "c": "\u0441", "x": "\u0445"}
LEET = {"a": "@", "o": "0", "i": "1", "e": "3", "s": "$"}


def homoglyph(t: str, r: random.Random, p: float = 0.5) -> str:
    """Swap Latin letters for Cyrillic look-alikes."""
    return "".join(HOMO[c] if c in HOMO and r.random() < p else c for c in t)


def zero_width(t: str, r: random.Random, p: float = 0.3) -> str:
    """Insert zero-width spaces inside words."""
    return "".join(c + (ZW if c.isalpha() and r.random() < p else "") for c in t)


def leet(t: str, r: random.Random, p: float = 0.5) -> str:
    """Substitute letter look-alikes for digits and symbols."""
    return "".join(LEET[c] if c in LEET and r.random() < p else c for c in t.lower())


def spacing(t: str, r: random.Random, p: float = 0.2) -> str:
    """Split long words with an interior space."""
    return " ".join(
        w[:1] + " " + w[1:] if len(w) > 3 and r.random() < p else w for w in t.split()
    )


def fullwidth(t: str, r: random.Random, p: float = 0.5) -> str:
    """Map ASCII to full-width forms."""
    return "".join(chr(ord(c) + 0xFEE0) if 33 <= ord(c) <= 126 and r.random() < p else c for c in t)


ATTACKS = {
    "homoglyph": homoglyph,
    "zero_width": zero_width,
    "leet": leet,
    "spacing": spacing,
    "fullwidth": fullwidth,
}


def eval_attack(
    predict_proba,
    spam_texts,
    attack,
    tau: float,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> tuple[float, float]:
    """Mean and std of spam recall under one attack, over seeded draws."""
    recalls = []
    for seed in seeds:
        rng = random.Random(seed)
        perturbed = [attack(t, rng) for t in spam_texts]
        recalls.append(float(np.mean(predict_proba(perturbed)[:, 1] >= tau)))
    return float(np.mean(recalls)), float(np.std(recalls))


def strip_urls(t: str) -> str:
    """E10: URL-masked ablation input."""
    return URL.sub(" ", t)


def url_slices(df):
    """E10 slices over a test frame: spam with a link, link-less spam, all ham."""
    return {
        "spam_with_url": df[(df["y"] == 1) & df["has_url"]],
        "spam_linkless": df[(df["y"] == 1) & ~df["has_url"]],
        "ham": df[df["y"] == 0],
    }


def augment_with_attacks(
    train: pd.DataFrame,
    families: tuple[str, ...] = ("leet", "zero_width"),
    copies: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """E9: add perturbed copies of the spam rows, one family at a time.

    Only positives are augmented, because the threat model is a spammer
    disguising spam. Ham is untouched, so augmentation cannot inflate the
    negative class with artefacts of the perturbing code.
    """
    unknown = [f for f in families if f not in ATTACKS]
    if unknown:
        raise ValueError(f"unknown attack families {unknown}")
    rng = random.Random(seed)
    positives = train[train["y"] == 1]
    extra = []
    for family in families:
        attack = ATTACKS[family]
        for _ in range(copies):
            part = positives.copy()
            part["text"] = [attack(t, rng) for t in positives["text"]]
            extra.append(part)
    if not extra:
        return train.reset_index(drop=True)
    return pd.concat([train, *extra], ignore_index=True)


def eval_families(
    predict_proba,
    spam_texts,
    tau: float,
    families: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    trained_on: tuple[str, ...] = (),
) -> pd.DataFrame:
    """E9: recall per attack family, grouped as in-family or cross-family.

    ``trained_on`` lists the families the model saw during augmentation; every
    other family is a held-out attack and its recall is the generalisation
    number that matters.
    """
    names = tuple(ATTACKS) if families is None else families
    rows = []
    for name in names:
        mean, std = eval_attack(predict_proba, spam_texts, ATTACKS[name], tau, seeds)
        rows.append(
            {
                "attack": name,
                "group": "in_family" if name in trained_on else "cross_family",
                "recall_mean": mean,
                "recall_std": std,
                "n": len(spam_texts),
            }
        )
    return pd.DataFrame(rows)


def family_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean recall per group, so in-family and cross-family are one number each."""
    return (
        table.groupby("group", as_index=False)
        .agg(recall_mean=("recall_mean", "mean"), attacks=("attack", "count"))
    )


def url_arm(predict_proba, df: pd.DataFrame, tau: float) -> pd.DataFrame:
    """E10: the same frame under the normal, URL-masked and link-less views.

    The URL-masked arm removes every link before scoring; the link-less slice
    keeps only spam that never had one. If recall on the link-less slice is far
    below the normal arm, the detector leans on links.
    """
    from smishlens.evaluate import metrics_at

    views = {
        "normal": df,
        "url_masked": df.assign(text=df["text"].map(strip_urls)),
        "spam_linkless": url_slices(df)["spam_linkless"],
    }
    rows = []
    for name, part in views.items():
        if part.empty:
            rows.append({"view": name, "n": 0, "n_pos": 0, "n_neg": 0, "recall": None, "fpr": None})
            continue
        p = predict_proba(part["text"].tolist())[:, 1]
        m = metrics_at(part["y"].values, p, tau)
        rows.append(
            {
                "view": name,
                "n": len(part),
                "n_pos": m["n_pos"],
                "n_neg": m["n_neg"],
                "recall": m["recall"],
                "fpr": m["fpr"],
            }
        )
    return pd.DataFrame(rows)
