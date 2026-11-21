"""The ONLY place a verdict becomes an action.

The LLM explainer never touches this. The rule is FPR-first: a missed advert
costs less than a blocked OTP or bank alert, so anything the model and the
conformal set disagree on is delivered rather than junked.
"""

from __future__ import annotations

ACTIONS = ("deliver", "junk", "quarantine")


def decide(p_spam: float, tau: float, conformal: list[int] | set[int]) -> str:
    """Map (probability, threshold, prediction set) to one of three actions.

    - ``quarantine``: the set is ambiguous ``{0, 1}`` or unlike either class ``{}``.
    - ``junk``: the set is ``{1}`` AND ``p_spam >= tau`` (both signals agree).
    - ``deliver``: everything else, including the disagree case.
    """
    S = set(conformal)
    if len(S) != 1:
        return "quarantine"
    if S == {1} and p_spam >= tau:
        return "junk"
    return "deliver"


def decide_ablation(p_spam: float, tau: float, conformal: list[int] | set[int]) -> str:
    """Alternative policy: disagreement quarantines instead of delivering.

    Reported as an ablation (E18), never as the default, because it raises the
    quarantine volume on ordinary messages.
    """
    S = set(conformal)
    if len(S) != 1:
        return "quarantine"
    if S == {1} and p_spam >= tau:
        return "junk"
    if S == {1}:
        return "quarantine"
    return "deliver"
