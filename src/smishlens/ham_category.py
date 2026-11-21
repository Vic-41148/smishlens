"""Coarse rule-based ham categories used by F1.

Rules are the ones recorded in the pool build log: first match wins,
bank-OTP alert -> transactional receipt -> promo -> chatty. They are coarse
buckets for counting false positives, not labels. No text leaves this module.
"""

from __future__ import annotations

import re

ORDER = ("bank_otp", "transactional", "promo", "chatty")


def _word(pat: str) -> re.Pattern:
    return re.compile(r"(?<!\w)(?:" + pat + r")(?!\w)", re.I)


RULES = {
    "bank_otp": _word(
        r"otp|otps|one[\s-]?time(?:\s+password)?|verification code|security code|"
        r"passcode|pin|do not share|never share|valid for|expires? in|will expire"
    ),
    "transactional": _word(
        r"receipt|invoice|order|payment|paid|debit(?:ed)?|credit(?:ed)?|balance|"
        r"transaction|txn|reference|ref no|amount|successfully|confirmed|delivery|"
        r"parcel|dispatched|booked|booking|ticket|statement|account|card|bank"
    ),
    "promo": _word(
        r"offer|discount|sale|deal|win|won|prize|cashback|coupon|subscribe|"
        r"unsubscribe|free|limited|promo|voucher|buy|shop|gift|bonus|reward"
    ),
}


def classify(text: object) -> str:
    s = str(text)
    for name in ("bank_otp", "transactional", "promo"):
        if RULES[name].search(s):
            return name
    return "chatty"
