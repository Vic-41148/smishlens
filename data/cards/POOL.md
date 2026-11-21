# DATA CARD — modern dev pool (re-frozen 2026-11-21 after the label audit)

The pool is built by `data/build_modern_pool.py` from the adapters in
`src/smishlens/sources.py` and frozen once at seed 42. Per-source cards live beside this file.

| Field | Value |
|---|---|
| Sources in the pool | `imc25`, `mishra`, `moz`, `smishtank`, `exais` (**ham-only**) |
| Sources held out of the pool | `wspr` (recall-only OOD), `africa` (synthetic) |
| Rows | 30,447 |
| Prevalence | 0.8527 (25,963 positive / 4,484 ham) |
| Dataset hash | `0bb9f4c85b6760d6` (supersedes `4b202c8f6ea707da`) |
| Near-duplicate threshold | 0.90 char 3–4-gram TF cosine (`use_idf=False`, `min_df=1`), plus an 8-row audited exclusion list |

## Splits (64 / 16 / 20, stratified by source and label)

| Split | Rows | Positives | Ham | Prevalence |
|---|---|---|---|---|
| train | 19,487 | 16,616 | 2,871 | 0.8527 |
| cal | 4,870 | 4,153 | 717 | 0.8528 |
| test | 6,090 | 5,194 | 896 | 0.8529 |

`cal` is used for conformal quantiles and temperature scaling only. `test` is touched once.

## Read this before quoting a number

**The pool is 85.3% positive.** IMC'25 is positives-only and contributes 24,462 rows, so the base
rate is nowhere near field prevalence. Consequences:

- Raw precision is meaningless here. Report **PPV at a realistic prevalence** (the Tier 1
  `ppv_by_prevalence` table is the pattern), never `precision` from the test split alone.
- PR-AUC is prevalence-sensitive, so an E6 "UCI vs modern pool" comparison must be read as a
  comparison of ranking quality and recall at a fixed FPR, not as a like-for-like AP.
- Accuracy is not reported anywhere in this project, and it would be especially misleading here.

This is a property of the available data, not a tuning choice: no source was subsampled or
capped, because discarding positives would silently change what the pool is. ExAIS is the one
deliberate exception: only its original ham rows are kept (see below).

## Where the numbers come from

Every count in this card and the per-source cards is read from
`data/processed/modern/data_audit.json` (written by `data/build_modern_pool.py`), except the raw
row/label counts, which `python data/audit_sources.py` prints from `data/raw/`. Nothing here is
estimated.

## What is committed and what is not

- Committed: `split_manifest.csv` (row id, source, split, labels, 16-hex text hash, no text),
  `loso_folds.json`, `data_audit.json`, `label_overrides.json` (the 8 label-audit corrections)
  and `dedup_exclusions.json` (8 audited near-duplicate row ids, no text).
- **Not committed:** `train.csv`, `cal.csv`, `test.csv`. MOZ and Smishtank have unclear
  licences, so their text stays local. Rebuild with `make data`.
- The manifest hashes are not a privacy control: a 16-hex SHA-256 of a short message can be
  recovered by guessing. They exist so the split can be audited, not to protect the text.

## Overlap handling

Order: label conflicts → exact dedup → normalised dedup → **audited exclusions** → near-dup vs
UCI → near-dup across sources (earlier source wins). Exact and near overlaps with UCI are removed
so the anchor and the pool never share a message. Stage counts are in `data_audit.json`.

The 0.90 rule is corpus-independent by construction: the vectoriser is fitted on only the two
frames being compared, with no idf and no document-frequency pruning. A pairwise audit found 36
Mishra rows that are near-duplicates of a UCI row; 28 clear 0.90 and 8 score 0.860–0.898. Those 8
are listed in `dedup_exclusions.json` and applied at the UCI-overlap stage, so all 36 leave the
pool. The list is audited, not exhaustive: other near-duplicates in the 0.86–0.90 band may remain
unaudited. Do not widen the list from another source without a new audit.

## LOSO folds

`loso_folds.json` defines five folds: `moz` and `mishra` have both classes; `imc25` and
`smishtank` are recall-only; `exais` is **FPR-only** (ham-only by construction, so it trains as a
negative source but only its false-positive rate is meaningful). Report the both-class macro
average with n = 2 and the per-source table beside it; never fold an FPR-only or recall-only row
into that macro. Train membership is a rule over the manifest
(`split == 'train' and source != held_out`), and only the eval ids are listed, so a fold can be
reconstructed without guessing.

## Why ExAIS is ham-only (2026-11-21)

The label audit found 30% binary disagreement on ExAIS (Wilson 21.9%–39.6%). The pattern is a
class-definition mismatch, not row noise: 27 of the 30 were benign quotes and operator/bank
notices stored as "spam", and 3 were genuine promos stored as ham. ExAIS is therefore never a
positive source: every original spam row is dropped before dedup and no ExAIS label is
overridden. Its 2,330 surviving ham rows are real Nigerian-English SMS and the only
bank/service-notification negatives the pool has.
