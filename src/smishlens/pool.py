"""Freeze the modern dev pool (Tier 2, Day 8).

Pipeline: adapters -> label-conflict drop -> exact dedup -> normalised dedup ->
near-duplicate removal against UCI and across sources -> 64/16/20 split
stratified by (source, y) -> LOSO fold definitions -> audit + manifest.

The split is frozen by ``data/build_modern_pool.py`` and must not be re-drawn.
Processed text is written to ``data/processed/modern/`` (git-ignored for the
unclear-licence sources); only the manifest, fold definitions and audit are
committed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from smishlens.data import SCHEMA, clean_dedup, dataset_hash, split_pool
from smishlens.normalise import normalise
from smishlens.sources import LOADERS, load_uci_from_raw

# Sources that form the pool, in dedup-priority order (first wins a duplicate).
# ExAIS was dropped on 2026-11-21: its "spam" class is bulk/unsolicited, not
# smishing (label audit: 30% binary disagreement, Wilson 21.9-39.6%, 27 of 30
# being benign quotes and operator notices). See data/audit and the log.
POOL_SOURCES = ("imc25", "mishra", "moz", "smishtank")
# Available but deliberately outside the pool: WSPR is a 68k recall-only OOD
# set and the Africa set is fully synthetic (roadmap §4.1: never in a test set).
OOD_SOURCES = ("wspr", "africa")
DROPPED_SOURCES = {"exais": "class-definition mismatch: bulk/benign positives, not row noise"}
BOTH_CLASSES = ("moz", "mishra")

NEAR_DUP_THRESHOLD = 0.90
SEED = 42


def load_pool(sources=POOL_SOURCES, raw: Path | None = None) -> pd.DataFrame:
    frames = [LOADERS[src](raw=raw) for src in sources]
    frame = pd.concat(frames, ignore_index=True)
    frame["synthetic"] = frame["synthetic"].fillna(False).astype(bool)
    return frame[SCHEMA]


def row_id(source: str, text: str) -> str:
    """Stable id that survives reordering: no text is exposed."""
    return hashlib.sha256(f"{source}\x1f{text}".encode()).hexdigest()[:16]


def _normalised_key(text: str) -> str:
    return " ".join(normalise(str(text), mask=True).split()).lower()


def _pairs_above(mat_a, mat_b, thr: float, chunk: int = 256) -> np.ndarray:
    """Rows of A with any cosine >= thr against B. Returns a bool mask over A."""
    keep = np.ones(mat_a.shape[0], dtype=bool)
    if mat_b.shape[1] == 0:
        return keep
    mat_b = mat_b.T.tocsc()
    for start in range(0, mat_a.shape[0], chunk):
        sims = (mat_a[start : start + chunk] @ mat_b).toarray()
        keep[start : start + chunk] = sims.max(axis=1) < thr
    return keep


def near_dup_mask(
    frame: pd.DataFrame, ref: pd.DataFrame, thr: float = NEAR_DUP_THRESHOLD
) -> np.ndarray:
    """Bool mask over ``frame``: True where no ``ref`` row is within ``thr`` cosine."""
    if ref.empty or frame.empty:
        return np.ones(len(frame), dtype=bool)
    docs = pd.concat([ref["text"], frame["text"]]).astype(str).tolist()
    # min_df=2 keeps the matrix small on the real pool, but on a handful of
    # documents it would collapse the vocabulary to the few shared n-grams and
    # make unrelated texts look identical.
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 4),
        min_df=2 if len(docs) >= 1000 else 1,
        sublinear_tf=True,
    )
    vec.fit(docs)
    mat_frame = vec.transform(frame["text"].astype(str))
    mat_ref = vec.transform(ref["text"].astype(str))
    return _pairs_above(mat_frame, mat_ref, thr)


def drop_near_dups(
    frame: pd.DataFrame, ref: pd.DataFrame, thr: float = NEAR_DUP_THRESHOLD
) -> tuple[pd.DataFrame, int]:
    """Drop rows of ``frame`` whose char 3-4-gram cosine to any ``ref`` row >= thr."""
    keep = near_dup_mask(frame, ref, thr)
    return frame[keep].reset_index(drop=True), int((~keep).sum())


def dedup_within(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Exact (case/space) then normalised (NFKC/confusables/URL-mask) dedup.

    Rows keep the earliest source in ``POOL_SOURCES`` order. Label conflicts are
    counted separately because a conflict drops *both* rows, which changes the
    class balance and must be visible in the audit.
    """
    before = len(frame)
    keys = frame["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    # Count distinct labels per text, not distinct texts per group.
    conflicts = frame.groupby(keys)["y"].transform("nunique") > 1
    n_conflict_rows = int(conflicts.sum())
    n_conflict_texts = int(keys[conflicts].nunique())
    rows_in_src = frame["source"].value_counts()
    conflict_src = frame.loc[conflicts, "source"].value_counts()
    frame = clean_dedup(frame, priority=list(POOL_SOURCES))
    after_exact = len(frame)
    exact_src = frame["source"].value_counts()
    norm_keys = frame["text"].map(_normalised_key)
    dup = norm_keys.duplicated()
    norm_src = frame.loc[dup, "source"].value_counts()
    frame = frame.loc[~dup].reset_index(drop=True)
    after_norm = len(frame)
    per_source = {}
    for src in POOL_SOURCES:
        n_in = int(rows_in_src.get(src, 0))
        n_conflict = int(conflict_src.get(src, 0))
        n_exact = int(exact_src.get(src, 0))
        n_norm = int(norm_src.get(src, 0))
        per_source[src] = {
            "rows_in": n_in,
            "dropped_label_conflict": n_conflict,
            "dropped_exact_duplicate": n_in - n_conflict - n_exact,
            "dropped_normalised": n_norm,
            "rows_out": n_exact - n_norm,
        }
    return frame, {
        "rows_in": before,
        "label_conflict_texts": n_conflict_texts,
        "dropped_label_conflict": n_conflict_rows,
        "dropped_exact_duplicate": before - n_conflict_rows - after_exact,
        "dropped_normalised": after_exact - after_norm,
        "rows_out": after_norm,
        "per_source": per_source,
    }


def drop_uci_overlap(
    frame: pd.DataFrame, uci: pd.DataFrame, thr: float = NEAR_DUP_THRESHOLD
) -> tuple[pd.DataFrame, dict]:
    """Remove modern rows that duplicate the anchor, exact and near."""
    uci_keys = set(uci["text"].map(_normalised_key))
    exact = frame["text"].map(_normalised_key).isin(uci_keys)
    rows_in_src = frame["source"].value_counts()
    exact_src = frame.loc[exact, "source"].value_counts()
    frame = frame[~exact].reset_index(drop=True)
    keep = near_dup_mask(frame, uci, thr)
    near_src = frame.loc[~keep, "source"].value_counts()
    frame = frame[keep].reset_index(drop=True)
    per_source = {
        src: {
            "rows_in": int(rows_in_src.get(src, 0)),
            "exact": int(exact_src.get(src, 0)),
            "near": int(near_src.get(src, 0)),
        }
        for src in POOL_SOURCES
    }
    return frame, {
        "exact": int(exact.sum()),
        "near": int((~keep).sum()),
        "rows_out": len(frame),
        "per_source": per_source,
    }


def dedup_across_sources(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Sequential near-dup removal in priority order: earlier sources win."""
    kept = pd.DataFrame(columns=frame.columns)
    log: dict[str, dict[str, int]] = {}
    for src in POOL_SOURCES:
        sub = frame[frame["source"] == src].reset_index(drop=True)
        if sub.empty:
            continue
        sub, dropped = drop_near_dups(sub, kept)
        log[src] = {"rows": int(len(sub) + dropped), "dropped": dropped}
        kept = pd.concat([kept, sub], ignore_index=True)
    return kept.reset_index(drop=True), log


def make_manifest(frame: pd.DataFrame, split_of: pd.Series) -> pd.DataFrame:
    """Committed, text-free record of the frozen split."""
    out = pd.DataFrame(
        {
            "row_id": [row_id(s, t) for s, t in zip(frame["source"], frame["text"], strict=True)],
            "source": frame["source"],
            "split": split_of.to_numpy(),
            "y": frame["y"].astype(int),
            "y3": frame["y3"],
            "lang": frame["lang"],
            "synthetic": frame["synthetic"].astype(bool),
            "text_sha256_16": [
                hashlib.sha256(str(t).encode()).hexdigest()[:16] for t in frame["text"]
            ],
            "has_url": frame["has_url"].astype(bool),
        }
    )
    if out["row_id"].duplicated().any():
        raise ValueError("row_id collision: two rows share (source, text)")
    return out


def make_loso_folds(frame: pd.DataFrame, split_of: pd.Series) -> dict:
    """LOSO folds: train on the pool train rows of the other sources.

    Both-class sources are evaluated on every row of the held-out source;
    positives-only sources are recall-only folds. Train membership is not
    stored (it is derivable from the manifest): only the eval ids are listed,
    so a fold can be reconstructed without guessing.
    """
    ids = pd.Series(
        [row_id(s, t) for s, t in zip(frame["source"], frame["text"], strict=True)],
        index=frame.index,
    )
    folds = []
    for src in POOL_SOURCES:
        held = frame["source"] == src
        train_mask = (frame["source"] != src) & (split_of == "train")
        folds.append(
            {
                "fold": f"heldout_{src}",
                "held_out": src,
                "kind": "both_classes" if src in BOTH_CLASSES else "recall_only",
                "train_rule": f"split == 'train' and source != '{src}'",
                "n_train": int(train_mask.sum()),
                "n_eval": int(held.sum()),
                "eval_positives": int(frame.loc[held, "y"].sum()),
                "eval_ids": sorted(ids[held].tolist()),
            }
        )
    return {
        "seed": SEED,
        "definition": (
            "train = split_manifest rows with split == 'train' and source != held_out; "
            "eval = every pool row of the held-out source, at its frozen split label"
        ),
        "folds": folds,
    }


def ood_stats(raw: Path | None = None) -> dict[str, dict]:
    """Counts for the sources deliberately kept outside the pool (quoted in the cards)."""
    out = {}
    for src in OOD_SOURCES:
        part = LOADERS[src](raw=raw)
        out[src] = {
            "rows": int(len(part)),
            "positives": int(part["y"].sum()),
            "ham": int((part["y"] == 0).sum()),
            "has_url": int(part["has_url"].sum()),
            "synthetic": bool(part["synthetic"].all()),
            "languages": part["lang"].value_counts().head(5).to_dict(),
        }
    return out


def apply_label_overrides(frame: pd.DataFrame, overrides: dict[str, str]) -> pd.DataFrame:
    """Apply ``{row_id: y3}`` corrections to a freshly loaded frame.

    Corrections are applied before dedup so a fixed label can also resolve a
    label conflict. Unknown ids raise: a stale review file must not silently
    become a no-op.
    """
    if not overrides:
        return frame
    frame = frame.copy()
    ids = [row_id(s, t) for s, t in zip(frame["source"], frame["text"], strict=True)]
    hit = 0
    for i, rid in enumerate(ids):
        y3 = overrides.get(rid)
        if y3 is None:
            continue
        if y3 not in ("ham", "spam", "smish"):
            raise ValueError(f"row {rid}: bad label {y3!r}")
        frame.at[i, "y3"] = y3
        frame.at[i, "y"] = 0 if y3 == "ham" else 1
        hit += 1
    missing = sorted(set(overrides) - set(ids))
    if missing:
        raise ValueError(f"{len(missing)} override ids not in the pool: {missing[:5]}")
    if hit != len(overrides):
        raise ValueError("override application mismatch")
    return frame


def write_pool(result: dict, out: Path) -> None:
    """Write the processed splits, text-free manifest, folds and audit into ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frame, split_of = result["frame"], result["split_of"]
    for name in ("train", "cal", "test"):
        part = frame[split_of == name]
        part[SCHEMA].to_csv(
            out / f"{name}.csv", index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
        )
    result["manifest"].to_csv(
        out / "split_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
    )
    (out / "loso_folds.json").write_text(json.dumps(result["loso"], indent=2) + "\n")
    (out / "data_audit.json").write_text(json.dumps(result["audit"], indent=2) + "\n")


def build(
    raw: Path | None = None,
    seed: int = SEED,
    label_overrides: dict[str, str] | None = None,
) -> dict:
    """Return the frozen pool, its splits, folds, manifest and audit."""
    raw_frame = load_pool(raw=raw)
    raw_frame = apply_label_overrides(raw_frame, label_overrides or {})
    uci = load_uci_from_raw(raw)

    frame, within = dedup_within(raw_frame)
    stages = {"raw": raw_frame["source"].value_counts().to_dict()}
    stages["after_within_dedup"] = frame["source"].value_counts().to_dict()
    frame, uci_overlap = drop_uci_overlap(frame, uci)
    stages["after_uci_overlap"] = frame["source"].value_counts().to_dict()
    frame, cross = dedup_across_sources(frame)
    stages["final"] = frame["source"].value_counts().to_dict()

    train, cal, test = split_pool(frame, seed=seed)
    split_of = pd.Series("train", index=frame.index)
    # split_pool resets indexes, so map back by row_id.
    ids = pd.Series(
        [row_id(s, t) for s, t in zip(frame["source"], frame["text"], strict=True)],
        index=frame.index,
    )
    lookup = {}
    for name, part in (("cal", cal), ("test", test)):
        for s, t in zip(part["source"], part["text"], strict=True):
            lookup[row_id(s, t)] = name
    split_of = ids.map(lookup).fillna("train")

    manifest = make_manifest(frame, split_of)
    folds = make_loso_folds(frame, split_of)

    per_source = {}
    for src in POOL_SOURCES:
        mask = frame["source"] == src
        part = frame[mask]
        per_source[src] = {
            "rows": int(mask.sum()),
            "ham": int((part["y"] == 0).sum()),
            "positives": int((part["y"] == 1).sum()),
            "languages": part["lang"].value_counts().head(5).to_dict(),
            "near_dup_dropped_vs_earlier_sources": cross.get(src, {}).get("dropped", 0),
        }

    split_counts = {}
    for name in ("train", "cal", "test"):
        mask = split_of == name
        n = int(mask.sum())
        pos = int(frame.loc[mask, "y"].sum())
        split_counts[name] = {
            "rows": n,
            "positives": pos,
            "ham": n - pos,
            "prevalence": round(pos / n, 6) if n else 0.0,
            "per_source": frame.loc[mask, "source"].value_counts().to_dict(),
        }

    audit = {
        "seed": seed,
        "label_overrides": {
            "n": len(label_overrides or {}),
            "applied_before_dedup": bool(label_overrides),
        },
        "pool_sources": list(POOL_SOURCES),
        "ood_only_sources": list(OOD_SOURCES),
        "excluded_sources": {
            "exais": "dropped 2026-11-21 after the label audit: its 'spam' class is bulk/unsolicited "
            "(benign quotes, operator notices), 30% binary disagreement (Wilson 21.9-39.6%); "
            "a class-definition mismatch, not row noise",
            "africa": "fully synthetic (is_synthetic=1 on every row); roadmap §4.1 forbids "
            "synthetic rows in a headline test set",
            "wspr": "68,029 recall-only phishing messages; kept as an OOD source so it "
            "cannot swamp the pool's class balance",
        },
        "dedup": {"within": within, "vs_uci": uci_overlap, "cross_source": cross},
        "per_source_stages": {
            stage: {k: int(v) for k, v in sorted(counts.items())}
            for stage, counts in stages.items()
        },
        "per_source": per_source,
        "ood_sources": ood_stats(raw),
        "splits": split_counts,
        "pool_rows": int(len(frame)),
        "pool_prevalence": round(float(frame["y"].mean()), 6),
        "near_dup_threshold": NEAR_DUP_THRESHOLD,
        "dataset_sha256_16": dataset_hash(frame),
        "notes": [
            "Splits are stratified by (source, y) with seed 42 and are frozen; do not redraw.",
            "Prevalence is far above field prevalence because the pool is positive-heavy by "
            "construction (IMC'25 is positives-only). Report PPV at realistic prevalence, "
            "never raw precision.",
            "The split is committed only as a text-free manifest; processed text stays "
            "git-ignored for the unclear-licence sources.",
        ],
    }
    return {
        "frame": frame,
        "split_of": split_of,
        "train": train,
        "cal": cal,
        "test": test,
        "manifest": manifest,
        "loso": folds,
        "audit": audit,
        "uci": uci,
    }
