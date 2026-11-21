"""Cross-source near-duplicate detection.

Exact dedup lives in smishlens.data.clean_dedup; this module catches the
Mishra->UCI, NUS->UCI and "combined" Hugging Face overlaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from smishlens.normalise import normalise


def near_dups(
    a_texts: list[str], b_texts: list[str], thr: float = 0.90, chunk: int = 256
) -> pd.DataFrame:
    """Pairs (i in A, j in B) with char 3-4-gram TF-IDF cosine >= thr.

    O(|A||B|): fine to roughly 20k x 20k. Past that use MinHash/LSH.
    """
    a = [normalise(t, mask=True) for t in a_texts]
    b = [normalise(t, mask=True) for t in b_texts]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 4), min_df=1, sublinear_tf=True)
    vec.fit(a + b)
    mat_a = vec.transform(a)
    mat_b = vec.transform(b).T.tocsc()
    rows: list[tuple[int, int, float]] = []
    for start in range(0, mat_a.shape[0], chunk):
        sims = (mat_a[start : start + chunk] @ mat_b).toarray()
        for i, j in zip(*np.where(sims >= thr), strict=True):
            rows.append((start + i, int(j), float(sims[i, j])))
    return pd.DataFrame(rows, columns=["i_a", "j_b", "sim"])


def drop_overlap(df: pd.DataFrame, ref_source: str, thr: float = 0.90):
    """Remove rows of every OTHER source that near-duplicate a row of ref_source.

    Returns (filtered frame, log dict) so the drop counts can be reported.
    """
    ref = df[df["source"] == ref_source]
    keep = pd.Series(True, index=df.index)
    log: dict[str, dict[str, int]] = {}
    for src in df["source"].unique():
        if src == ref_source:
            continue
        sub = df[df["source"] == src]
        pairs = near_dups(sub["text"].tolist(), ref["text"].tolist(), thr)
        drop_index = sub.index[pairs["i_a"].unique()] if len(pairs) else []
        keep.loc[drop_index] = False
        log[str(src)] = {"rows": len(sub), "dropped": int(len(drop_index))}
    return df[keep].reset_index(drop=True), log
