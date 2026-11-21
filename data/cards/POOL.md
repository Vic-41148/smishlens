# DATA CARD — modern dev pool (frozen 2026-11-21)

The pool is built by `data/build_modern_pool.py` from the adapters in
`src/smishlens/sources.py` and frozen once at seed 42. Per-source cards live beside this file.

| Field | Value |
|---|---|
| Sources in the pool | `imc25`, `mishra`, `exais`, `moz`, `smishtank` |
| Sources held out of the pool | `wspr` (recall-only OOD), `africa` (synthetic) |
| Rows | 31,931 |
| Prevalence | 0.8601 (27,463 positive / 4,468 ham) |
| Dataset hash | `4015672722968db6` |
| Near-duplicate threshold | 0.90 char 3–4-gram TF-IDF cosine |

## Splits (64 / 16 / 20, stratified by source and label)

| Split | Rows | Positives | Ham | Prevalence |
|---|---|---|---|---|
| train | 20,435 | 17,575 | 2,860 | 0.8600 |
| cal | 5,108 | 4,393 | 715 | 0.8600 |
| test | 6,388 | 5,495 | 893 | 0.8602 |

`cal` is used for conformal quantiles and temperature scaling only. `test` is touched once.

## Read this before quoting a number

**The pool is 86% positive.** IMC'25 is positives-only and contributes 24,462 rows, so the base
rate is nowhere near field prevalence. Consequences:

- Raw precision is meaningless here. Report **PPV at a realistic prevalence** (the Tier 1
  `ppv_by_prevalence` table is the pattern), never `precision` from the test split alone.
- PR-AUC is prevalence-sensitive, so an E6 "UCI vs modern pool" comparison must be read as a
  comparison of ranking quality and recall at a fixed FPR, not as a like-for-like AP.
- Accuracy is not reported anywhere in this project, and it would be especially misleading here.

This is a property of the available data, not a tuning choice: no source was subsampled or
capped, because discarding positives would silently change what the pool is.

## Where the numbers come from

Every count in this card and the per-source cards is read from
`data/processed/modern/data_audit.json` (written by `data/build_modern_pool.py`), except the raw
row/label counts, which `python data/audit_sources.py` prints from `data/raw/`. Nothing here is
estimated.

## What is committed and what is not

- Committed: `split_manifest.csv` (row id, source, split, labels, 16-hex text hash, no text),
  `loso_folds.json`, `data_audit.json`.
- **Not committed:** `train.csv`, `cal.csv`, `test.csv`. MOZ, ExAIS and Smishtank have unclear
  licences, so their text stays local. Rebuild with `make data`.
- The manifest hashes are not a privacy control: a 16-hex SHA-256 of a short message can be
  recovered by guessing. They exist so the split can be audited, not to protect the text.

## Overlap handling

Order: label conflicts → exact dedup → normalised dedup → near-dup vs UCI → near-dup across
sources (earlier source wins). Exact and near overlaps with UCI are removed so the anchor and the
pool never share a message. Stage counts are in `data_audit.json`.

## LOSO folds

`loso_folds.json` defines five folds: `moz`, `exais` and `mishra` have both classes;
`imc25` and `smishtank` are recall-only. Train membership is a rule over the manifest
(`split == 'train' and source != held_out`), and only the eval ids are listed, so a fold can be
reconstructed without guessing.
