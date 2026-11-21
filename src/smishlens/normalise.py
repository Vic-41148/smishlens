"""Unicode and formatting normaliser.

Its own module so every experiment can run it ON and OFF. Confusables are
folded only inside tokens that also contain an ASCII letter ("mixed" mode): a
global fold would turn genuine Russian or Ukrainian words into Latin garbage,
which matters because the pool is multilingual.
"""

from __future__ import annotations

import re
import unicodedata

CONFUSABLES = {
    "\u0430": "a",  # Cyrillic a
    "\u0435": "e",  # Cyrillic ie
    "\u043e": "o",  # Cyrillic o
    "\u0440": "p",  # Cyrillic er
    "\u0441": "c",  # Cyrillic es
    "\u0445": "x",  # Cyrillic ha
    "\u0443": "y",  # Cyrillic u
    "\u0456": "i",  # Cyrillic dotted i
    "\u0455": "s",  # Cyrillic dze
    "\u03bf": "o",  # Greek omicron
    "\u03b1": "a",  # Greek alpha
}


def load_confusables(path: str) -> dict[str, str]:
    """Parse a Unicode UTS #39 confusables.txt file.

    Keeps ONLY non-ASCII sources: the full skeleton also maps ASCII characters
    (I->l, 1->l) and would corrupt ordinary text.
    """
    table: dict[str, str] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            if line.startswith("#") or ";" not in line:
                continue
            src, dst = (field.strip() for field in line.split(";")[:2])
            if " " in src:
                continue
            ch = chr(int(src, 16))
            if ord(ch) < 128:
                continue
            table[ch] = "".join(chr(int(h, 16)) for h in dst.split())
    return table


# hxxp:// and [.] are deliberate defanging used to slip past naive URL checks.
URL = re.compile(
    r"(?:h(?:tt|xx)ps?://|www\.)\S+"
    r"|\b[\w-]+(?:\[\.\]|\(\.\))\S*"
    r"|\b[\w-]+\.(?:com|net|org|in|co|uk|xyz|top|info|link|ly)\b\S*",
    re.I,
)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
MONEY = re.compile(r"[£$€₹]\s?\d[\d.,]*|\b\d[\d.,]*\s?(?:gbp|usd|eur|inr|rs|pence)\b", re.I)
PHONE = re.compile(r"\+?\d[\d\-\s()]{7,}\d")
SHORT = re.compile(r"\b\d{5,6}\b")
NUM = re.compile(r"\b\d+(?:[.,]\d+)?\b")
WS = re.compile(r"\s+")


def _fold(text: str, table: dict[str, str], mode: str) -> tuple[str, int]:
    """Fold look-alikes.

    mode="mixed" (default): map only inside tokens that ALSO contain an ASCII
    letter, so genuine Russian/Ukrainian/Greek words are left alone.
    mode="global": map everywhere (ablation arm).
    """
    if mode == "global":
        return "".join(table.get(c, c) for c in text), sum(c in table for c in text)
    pieces: list[str] = []
    count = 0
    for word in re.split(r"(\s+)", text):
        if any(c.isascii() and c.isalpha() for c in word):
            count += sum(c in table for c in word)
            word = "".join(table.get(c, c) for c in word)
        pieces.append(word)
    return "".join(pieces), count


def normalise_with_flags(
    text: object,
    mask: bool = True,
    table: dict[str, str] | None = None,
    mode: str = "mixed",
    lower: bool = True,
) -> tuple[str, dict[str, int]]:
    """Normalise one message and return it with a dict of disguise flags."""
    table = CONFUSABLES if table is None else table
    raw = str(text)

    fullwidth = sum(1 for c in raw if 0xFF01 <= ord(c) <= 0xFF5E)
    out = unicodedata.normalize("NFKC", raw)  # folds fullwidth, NOT cyrillic
    fmt = sum(1 for c in out if unicodedata.category(c) == "Cf")  # zero-width and other format chars
    out = "".join(c for c in out if unicodedata.category(c) != "Cf")
    out, confusables = _fold(out, table, mode)

    if mask:  # MASK, don't delete: the token type is a strong signal
        # EMAIL before URL: the URL pattern also matches a bare "b.com" inside
        # "a@b.com" and would leave a dangling "a@".
        out = EMAIL.sub(" <EMAIL> ", out)
        out = URL.sub(" <URL> ", out)
        out = MONEY.sub(" <MONEY> ", out)
        out = PHONE.sub(" <PHONE> ", out)
        out = SHORT.sub(" <SHORTCODE> ", out)
        out = NUM.sub(" <NUM> ", out)
    if lower:
        out = out.lower()

    flags = {
        key: value
        for key, value in (
            ("fullwidth", fullwidth),
            ("format_chars", fmt),
            ("confusables", confusables),
        )
        if value
    }
    return WS.sub(" ", out).strip(), flags


def normalise(
    text: object,
    mask: bool = True,
    table: dict[str, str] | None = None,
    mode: str = "mixed",
    lower: bool = True,
) -> str:
    """Normalise one message, discarding the flags."""
    return normalise_with_flags(text, mask, table, mode, lower)[0]


MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})
MIXED = re.compile(r"^(?=.*[a-z])(?=.*\d)[a-z\d]{3,}$")


def deleet_mixed(text: str) -> str:
    """Decode only mixed letter-digit tokens (fr33 -> free).

    Ablation flag only: it breaks words such as '4get'.
    """
    return " ".join(w.translate(MAP) if MIXED.match(w) else w for w in text.split())
