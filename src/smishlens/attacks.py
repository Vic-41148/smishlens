"""Standard published perturbation families, for evaluating and hardening THIS
detector only (RQ3, RQ6).

Ethics: this module exists to measure your own robustness and to show the
normaliser earns its place. Do not package it as a bypass tool and do not craft
new real-world phishing content.
"""

from __future__ import annotations

import random

import numpy as np

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
