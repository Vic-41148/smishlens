"""E7/E8 evasion and E10 URL ablation on the pool test split.

E7 = normaliser OFF at inference, E8 = normaliser ON. The classical svm keeps
its fitted vectorisers; the CleanText switch is flipped on a deep copy, so no
model is refit and the frozen threshold is unchanged. mmBERT-base is the frozen
E3 checkpoint; E7 feeds raw text, E8 uses the documented ``prep``.

E10 masks every URL, then scores the link-less spam slice separately. Output is
aggregates only: no message text.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from smishlens.attacks import ATTACKS, strip_urls
from smishlens.evaluate import metrics_at
from smishlens.features import CleanText, MetaFeatures
from smishlens.normalise import URL
from smishlens.pool_experiments import (
    RESULTS,
    _est,
    build_fold_matrices,
    load_pool,
)

E3_MMBERT = Path(__file__).resolve().parents[2] / "artifacts" / "mmbert" / "e3_base"
E3_CLASSICAL_JSON = RESULTS / "e3_pool.json"
E3_MMBERT_JSON = RESULTS / "e3_pool_mmbert_base.json"
BOTH_CLASSES = ("mishra", "moz")
SEEDS = (0, 1, 2)


def url_obfuscation(t: str, r: random.Random, p: float = 0.8) -> str:
    """Defang links: hxxp and [.] forms, which the normaliser's URL regex knows."""

    def repl(match: re.Match) -> str:
        if r.random() >= p:
            return match.group(0)
        url = re.sub(r"(?i)^https?://", "hxxp://", match.group(0))
        return url.replace(".", "[.]")

    return URL.sub(repl, t)


ATTACK_SET = {
    "spacing": ATTACKS["spacing"],
    "leet": ATTACKS["leet"],
    "homoglyph": ATTACKS["homoglyph"],
    "url_obfuscation": url_obfuscation,
}


def _clone_feats(feats, enabled: bool, strip: bool = False):
    """Flip CleanText/MetaFeatures switches on a fitted FeatureUnion copy."""
    out = copy.deepcopy(feats)
    for _, branch in out.transformer_list:
        for _, step in getattr(branch, "steps", []):
            if isinstance(step, CleanText):
                step.enabled = enabled
                step.strip_urls = strip
            elif isinstance(step, MetaFeatures):
                step.use_url = not strip
    return out


def _e3_reference(path: Path, model: str) -> dict:
    """Frozen E3 test macro at its own threshold, read from the results JSON."""
    data = json.loads(path.read_text())
    rows = [r for r in data["rows"] if r.get("model") == model]
    macro = next(r for r in rows if r.get("source") == "macro")
    tau = next(float(r["tau"]) for r in rows if "tau" in r)
    return {
        "tau": tau,
        "recall": float(macro["macro_recall"]),
        "fpr": float(macro["macro_fpr"]),
    }


def _both_class_macro(test: pd.DataFrame, p: np.ndarray, tau: float) -> tuple[float, float]:
    """E3 aggregation: unweighted mean over the both-class sources only."""
    values = []
    for src in BOTH_CLASSES:
        mask = (test["source"] == src).to_numpy()
        m = metrics_at(test.loc[mask, "y"].to_numpy(), np.asarray(p)[mask], tau)
        values.append((m["recall"], m["fpr"]))
    return float(np.mean([v[0] for v in values])), float(np.mean([v[1] for v in values]))


def _attack_recall(score_fn, texts, attack, tau: float) -> dict:
    recalls = []
    for seed in SEEDS:
        rng = random.Random(seed)
        perturbed = [attack(t, rng) for t in texts]
        recalls.append(float(np.mean(score_fn(perturbed) >= tau)))
    return {
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "n": len(texts),
    }


def _raw_mmbert_scores(scorer, texts) -> np.ndarray:
    """E7: tokenize raw text, skipping the normaliser in ``prep``."""
    import torch

    texts = list(texts)
    if not texts:
        return np.empty(0)
    scorer.model.eval()
    out = []
    with torch.no_grad(), torch.autocast(
        scorer.dev, dtype=torch.float16, enabled=(scorer.dev == "cuda")
    ):
        for i in range(0, len(texts), 32):
            enc = scorer.tok(
                texts[i : i + 32],
                truncation=True,
                max_length=scorer.max_len,
                padding=True,
                return_tensors="pt",
            ).to(scorer.dev)
            logits = scorer.model(**enc).logits.float()
            out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    margins = np.concatenate(out)
    return 1 / (1 + np.exp(-np.clip(margins / scorer.T, -40, 40)))


def run() -> dict:
    frame = load_pool()
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    if not pd.api.types.is_bool_dtype(test["has_url"]):
        raise TypeError("has_url must be a bool column for the E10 slice")
    y_all = test["y"].to_numpy()
    mask_spam = y_all == 1
    mask_ham = y_all == 0
    spam = test[mask_spam].reset_index(drop=True)
    spam_texts = spam["text"].tolist()

    mats = build_fold_matrices(frame[frame["split"] == "train"])
    est = _est("svm").fit(mats.X, mats.y)
    feats_off = _clone_feats(mats.feats, enabled=False)
    feats_on = _clone_feats(mats.feats, enabled=True)
    feats_masked = _clone_feats(mats.feats, enabled=True, strip=True)

    def svm_score(view):
        return lambda texts: est.decision_function(view.transform(texts))

    meta = json.loads((E3_MMBERT / "metadata.json").read_text())
    T = float(meta["calibration"]["T"])
    from smishlens.train_mmbert import MMBertScorer

    scorer = MMBertScorer.load(E3_MMBERT / "mmbert", T=T)

    refs = {
        "svm": _e3_reference(E3_CLASSICAL_JSON, "svm"),
        "mmbert_base": _e3_reference(E3_MMBERT_JSON, "mmbert_base"),
    }
    # The frozen E3 thresholds are the comparison point; the cached-matrix OOF
    # path shifts them by ~0.02 and cannot reproduce the E3 rows.
    if abs(refs["mmbert_base"]["tau"] - float(meta["threshold"]["value"])) > 1e-9:
        raise AssertionError("mmBERT E3 threshold disagrees with its metadata")

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

    identity = partial(url_obfuscation, p=0)
    out = {
        "scope": "pool test split; frozen models; thresholds unchanged",
        "note": (
            "E7 normaliser OFF, E8 normaliser ON; attacks applied to spam only; "
            "E10 masks every URL and scores link-less spam separately. "
            "Aggregates only."
        ),
        "seeds": list(SEEDS),
        "models": {},
    }

    for name, cfg in models.items():
        tau = cfg["tau"]
        ref = refs[name]
        block = {
            "tau": tau,
            "n_spam": int(mask_spam.sum()),
            "n_ham": int(mask_ham.sum()),
            "baseline": {},
            "attacks": {},
            "checks": {},
        }
        scores = {}
        for view, key in (("off", "score_off"), ("on", "score_on")):
            p_all = np.asarray(cfg[key](test["text"].tolist()), dtype=float)
            scores[view] = p_all
            m_spam = metrics_at(y_all[mask_spam], p_all[mask_spam], tau)
            m_ham = metrics_at(y_all[mask_ham], p_all[mask_ham], tau)
            block["baseline"][view] = {
                "recall": m_spam["recall"],
                "fpr": m_ham["fpr"],
            }
            block["attacks"][view] = {}
            for attack_name, attack in ATTACK_SET.items():
                block["attacks"][view][attack_name] = _attack_recall(
                    cfg[key], spam_texts, attack, tau
                )
            # An identity perturbation must reproduce the clean recall exactly.
            identity_recall = _attack_recall(cfg[key], spam_texts, identity, tau)[
                "recall_mean"
            ]
            if abs(identity_recall - m_spam["recall"]) > 1e-12:
                raise AssertionError(
                    f"{name}/{view}: identity attack changed recall "
                    f"({identity_recall} vs {m_spam['recall']})"
                )
            block["checks"][view] = {
                "identity_recall": identity_recall,
                "clean_recall": m_spam["recall"],
            }

        # E8 (normaliser ON, no attack) must reproduce the E3 test macro.
        observed_recall, observed_fpr = _both_class_macro(test, scores["on"], tau)
        if abs(observed_recall - ref["recall"]) > 1e-6:
            raise AssertionError(
                f"{name}: E3 recall mismatch {observed_recall} vs {ref['recall']}"
            )
        if abs(observed_fpr - ref["fpr"]) > 1e-6:
            raise AssertionError(f"{name}: E3 FPR mismatch {observed_fpr} vs {ref['fpr']}")
        block["e3_check"] = {
            "aggregation": "unweighted macro over mishra + moz test rows",
            "tau": tau,
            "expected": {"recall": ref["recall"], "fpr": ref["fpr"]},
            "observed": {"recall": observed_recall, "fpr": observed_fpr},
        }

        views = {}
        for view, texts in (
            ("normal", test["text"].tolist()),
            ("url_masked", [strip_urls(t) for t in test["text"].tolist()]),
        ):
            key = "score_masked" if view == "url_masked" else "score_on"
            p = cfg[key](texts)
            m = metrics_at(y_all, p, tau)
            views[view] = {
                "n": int(len(test)),
                "n_pos": m["n_pos"],
                "n_neg": m["n_neg"],
                "recall": m["recall"],
                "fpr": m["fpr"],
            }
        linkless = spam[~spam["has_url"]]
        p = cfg["score_on"](linkless["text"].tolist())
        m = metrics_at(linkless["y"].to_numpy(), p, tau)
        views["spam_linkless"] = {
            "n": int(len(linkless)),
            "n_pos": m["n_pos"],
            "n_neg": m["n_neg"],
            "recall": m["recall"],
            "fpr": m["fpr"],
        }
        block["e10"] = views
        out["models"][name] = block

    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "e7_e10_evasion.json"))
    args = ap.parse_args(argv)
    payload = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    for name, block in payload["models"].items():
        print(f"{name} baseline off {block['baseline']['off']} on {block['baseline']['on']}")
        chk = block["e3_check"]
        print(
            f"  E3 check {chk['observed']['recall']:.6f}/{chk['observed']['fpr']:.6f} "
            f"vs {chk['expected']['recall']:.6f}/{chk['expected']['fpr']:.6f}"
        )
        for view in ("off", "on"):
            for attack, row in block["attacks"][view].items():
                print(f"  {view} {attack}: {row['recall_mean']:.3f} +/- {row['recall_std']:.3f}")
        for view, row in block["e10"].items():
            print(f"  E10 {view}: n={row['n']} recall={row['recall']} fpr={row['fpr']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
