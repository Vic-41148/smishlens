"""Ten invented demo messages for the hosted app.

Every row is synthetic and written for this app. None is taken from gold v1/v2
or from the pool. ``verify_examples`` checks the 0.90 tf-only near-dup rule
against both and returns counts only; no message text is printed.
"""

from __future__ import annotations

EXAMPLES = [
    {
        "label": "Parcel fee",
        "text": (
            "Your parcel is waiting at the depot. Pay the 1.99 redelivery fee at "
            "hxxp://parcel-fee.example/pay before 6pm."
        ),
    },
    {
        "label": "Bank OTP (legit)",
        "text": "Your one-time code is 482913. Do not share it. It expires in 10 minutes.",
    },
    {
        "label": "Dinner plan",
        "text": "Hi, are we still on for dinner at 7? I can book the table.",
    },
    {
        "label": "Prize lure",
        "text": (
            "Congratulations! You have won a 500 voucher. Claim at "
            "hxxp://prize-claim.example now."
        ),
    },
    {
        "label": "Hinglish KYC",
        "text": (
            "Aapka bank account block ho gaya hai. KYC ke liye is link par click "
            "karein: hxxp://kyc-update.example"
        ),
    },
    {
        "label": "Invoice",
        "text": "Your invoice INV-2041 for 48.00 is attached. Payment is due on 30 Nov.",
    },
    {
        "label": "Meeting move",
        "text": "Meeting moved to 3pm in room 4. Bring the printed agenda.",
    },
    {
        "label": "Account threat",
        "text": (
            "Final warning: your account will be suspended. Verify now at "
            "hxxp://secure-verify.example/login"
        ),
    },
    {
        "label": "Family note",
        "text": "Mom, I'll be late tonight. Don't wait up for dinner.",
    },
    {
        "label": "Voicemail",
        "text": "You have 1 new voicemail. Call 0906 123 4567 to listen. Charges apply.",
    },
]


def verify_examples() -> dict:
    """Counts only: no text, no ids, no gold row is reproduced."""
    from smishlens.dedup import near_dups
    from smishlens.gold_v2 import GOLD_V1, GOLD_V2, read_gold
    from smishlens.pool_experiments import load_pool

    demo = [row["text"] for row in EXAMPLES]
    gold = read_gold(GOLD_V1)["text"].tolist() + read_gold(GOLD_V2)["text"].tolist()
    pool = load_pool()["text"].tolist()
    out = {"n_examples": len(demo), "threshold": 0.90}
    for name, ref in (("gold", gold), ("pool", pool)):
        # thr=0 keeps every scored pair so max_sim is a real maximum, not "0
        # because nothing crossed the threshold".
        pairs = near_dups(demo, ref, thr=0.0)
        flagged = pairs[pairs["sim"] >= 0.90] if len(pairs) else pairs
        out[name] = {
            "n_pairs": int(len(flagged)),
            "n_examples_with_match": int(flagged["i_a"].nunique()) if len(flagged) else 0,
            "max_sim": float(pairs["sim"].max()) if len(pairs) else 0.0,
        }
    return out


def main() -> int:
    import json

    from smishlens.pool_experiments import RESULTS

    payload = verify_examples()
    out = RESULTS / "demo_examples_check.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
