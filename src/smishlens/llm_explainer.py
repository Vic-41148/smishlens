"""OPTIONAL phrasing helper.

It never sees the raw SMS, never changes a verdict and never changes an action.
Only the label, the calibrated probability and up to six sanitised single words
reach the model. Any failure falls back to a deterministic sentence.
"""

from __future__ import annotations

import json
import re
import string
import urllib.request

from smishlens.normalise import normalise

WORD_OK = re.compile(r"^[a-z0-9]{2,20}$")
BANNED = {
    "ignore",
    "instruction",
    "instructions",
    "system",
    "prompt",
    "assistant",
    "override",
    "previous",
    "disregard",
}


def sanitise_words(words, k: int = 6) -> list[str]:
    """Attacker-controlled text -> at most ``k`` plain lowercase alphanumeric words."""
    out: list[str] = []
    for word in words:
        w = str(word).strip(string.punctuation + string.whitespace)
        if re.search(r"[^\w]", w):
            continue  # interior punctuation or whitespace (URLs, injected newlines)
        w = re.sub(r"[^a-z0-9]", "", normalise(w, mask=False))
        if WORD_OK.match(w) and w not in BANNED and w not in out:
            out.append(w)
        if len(out) >= k:
            break
    return out


TEMPLATE = (
    "Write ONE plain sentence (max 25 words) telling a non-expert why an SMS "
    "filter reacted to a message.\n"
    "Verdict: {label}. Confidence: {p:.2f}. Words that pushed the score up: {words}.\n"
    "Use only this information. Do not give instructions. Do not say the message "
    "is safe. No links."
)


def build_prompt(label: str, p: float, words) -> str:
    safe = ", ".join(sanitise_words(words)) or "none"
    return TEMPLATE.format(label=label, p=float(p), words=safe)


FORBID = re.compile(r"https?://|www\.|\b(safe|legitimate|genuine|not spam|ignore)\b", re.I)
THINK = re.compile(r"<think.*?</think(?:ing)?\s*>", re.S)


def guard_output(text: str | None, max_len: int = 240) -> str | None:
    """Accept one short line; reject links and safety claims."""
    t = THINK.sub("", text or "").strip()
    t = t.splitlines()[0].strip() if t else ""
    return t if t and len(t) <= max_len and not FORBID.search(t) else None


def fallback_sentence(words) -> str:
    safe = sanitise_words(words)
    if safe:
        return "Flagged mainly because of: " + ", ".join(safe) + "."
    return "The overall wording resembles known scam messages."


def ollama(
    prompt: str,
    model: str = "qwen3:4b-instruct",
    url: str = "http://localhost:11434/api/generate",
    timeout: int = 20,
) -> str:
    """Local backend. Stays on the machine."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 60},
        }
    ).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"]


def claude(prompt: str, model: str = "claude-haiku-5-5") -> str:
    """API backend. Sends words off-device; needs ``ANTHROPIC_API_KEY``."""
    import anthropic

    resp = anthropic.Anthropic().messages.create(
        model=model, max_tokens=80, messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text


_CACHE: dict = {}


def reset_cache() -> None:
    _CACHE.clear()


def explain(label, p, words, action, mode: str = "off", backend=None):
    """mode ``off`` | ``local`` | ``api``. Runs only for quarantined messages."""
    if mode == "off" or action != "quarantine":
        return None
    key = (label, round(float(p), 2), tuple(sanitise_words(words)), mode)
    if key in _CACHE:
        return _CACHE[key]
    fn = backend or {"local": ollama, "api": claude}.get(mode)
    if fn is None:
        return None
    try:
        out = guard_output(fn(build_prompt(label, p, words)))
    except Exception:  # a helper outage must never break the filter
        out = None
    _CACHE[key] = out or fallback_sentence(words)
    return _CACHE[key]
