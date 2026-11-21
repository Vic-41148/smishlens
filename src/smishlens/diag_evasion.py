"""Per-source breakdown of E7/E8/E10 spam recall.

Frozen models and thresholds; no refit. imc25 supplies about 94% of test spam,
so the pool-wide evasion number is dominated by it. Aggregates only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from smishlens.attacks import strip_urls
from smishlens.e7_e10 import (
    ATTACK_SET,
    E3_CLASSICAL_JSON,
    E3_MMBERT,
    E3_MMBERT_JSON,
    _attack_recall,
    _clone_feats,
    _e3_reference,
    _raw_mmbert_scores,
)
from smishlens.pool_experiments import RESULTS, _est, build_fold_matrices, load_pool

SOURCES = ("imc25", "mishra", "moz", "smishtank")


def _recall(score_fn, texts, tau: float) -> float:
    if not texts:
        return float("nan")
    return float(np.mean(np.asarray(score_fn(texts)) >= tau))


def _block(cfg: dict, test, tau: float) -> dict:
    spam = test[test["y"] == 1]
    out = {}
    for src in SOURCES:
        sub = spam[spam["source"] == src]
        texts = sub["text"].tolist()
        linkless = sub[~sub["has_url"]]["text"].tolist()
        block = {
            "n_spam": int(len(sub)),
            "baseline": {
                "off": _recall(cfg["score_off"], texts, tau),
                "on": _recall(cfg["score_on"], texts, tau),
            },
            "attacks": {"off": {}, "on": {}},
            "e10": {
                "normal": _recall(cfg["score_on"], texts, tau),
                "url_masked": _recall(cfg["score_masked"], texts, tau),
                "spam_linkless": _recall(cfg["score_on"], linkless, tau),
                "n_linkless": int(len(linkless)),
            },
        }
        for view, key in (("off", "score_off"), ("on", "score_on")):
            for name, attack in ATTACK_SET.items():
                block["attacks"][view][name] = _attack_recall(
                    cfg[key], texts, attack, tau
                )
        out[src] = block
    return out


def run() -> dict:
    frame = load_pool()
    test = frame[frame["split"] == "test"].reset_index(drop=True)

    mats = build_fold_matrices(frame[frame["split"] == "train"])
    est = _est("svm").fit(mats.X, mats.y)
    feats_off = _clone_feats(mats.feats, enabled=False)
    feats_on = _clone_feats(mats.feats, enabled=True)
    feats_masked = _clone_feats(mats.feats, enabled=True, strip=True)

    def svm_score(view):
        return lambda texts: est.decision_function(view.transform(texts))

    meta = json.loads((E3_MMBERT / "metadata.json").read_text())
    from smishlens.train_mmbert import MMBertScorer

    scorer = MMBertScorer.load(E3_MMBERT / "mmbert", T=float(meta["calibration"]["T"]))
    refs = {
        "svm": _e3_reference(E3_CLASSICAL_JSON, "svm"),
        "mmbert_base": _e3_reference(E3_MMBERT_JSON, "mmbert_base"),
    }
    models = {
        "svm": {
            "tau": refs["svm"]["tau"],
            "score_off": svm_score(feats_off),
            "score_on": svm_score(feats_on),
            "score_masked": svm_score(feats_masked),
        },
        "mmbert_base": {
            "tau": refs["mmbert_base"]["tau"],
            "score_off": lambda texts: _raw_mmbert_scores(scorer, texts),
            "score_on": lambda texts: scorer.predict_proba(list(texts))[:, 1],
            "score_masked": lambda texts: scorer.predict_proba(
                [strip_urls(t) for t in texts]
            )[:, 1],
        },
    }

    out = {
        "scope": "pool test split; frozen models; thresholds unchanged",
        "note": (
            "per-source spam recall for E7 (normaliser OFF), E8 (ON) and E10 "
            "(URL masked / link-less). imc25 is about 94% of test spam, so the "
            "pool-wide number is dominated by it. Aggregates only."
        ),
        "seeds": [0, 1, 2],
        "models": {},
    }
    for name, cfg in models.items():
        out["models"][name] = {
            "tau": cfg["tau"],
            "sources": _block(cfg, test, cfg["tau"]),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "e7_e10_by_source.json"))
    args = ap.parse_args(argv)
    payload = run()
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    for model, block in payload["models"].items():
        for src, row in block["sources"].items():
            print(
                f"{model} {src}: n={row['n_spam']} off={row['baseline']['off']:.4f} "
                f"on={row['baseline']['on']:.4f} linkless={row['e10']['spam_linkless']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
