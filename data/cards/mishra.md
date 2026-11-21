# DATA CARD — Mishra & Soni

| Field | Value |
|---|---|
| Source | Mendeley `f45bkkt8pr`, `Dataset_5971.csv` |
| Accessed | 2026-11-21 |
| Licence | CC BY 4.0 |
| Rows (raw) | 5,971 |
| Labels | 4,844 ham / 489 spam / 638 smishing (3-class) |
| Languages | English (not declared per row) |
| Rows kept in the pool | 599 (367 ham / 232 positive) |
| Role | Both-class LOSO source |

## What it is

A 3-class SMS set used in the smishing literature. Columns: `LABEL, TEXT, URL, EMAIL, PHONE`.

## Decisions taken in the loader

- **Label casing is normalised** (`spam`/`Spam`, `smishing`/`Smishing`); raw counts above are
  after normalisation.
- `y3` keeps the 3-class label; `y = 1` for both spam and smishing.
- The `URL`/`EMAIL`/`PHONE` flag columns are not used: they are derived from the text, so using
  them as features would leak the label.

## Known biases and the big loss of rows

This corpus **overlaps UCI heavily** (the roadmap estimates ~4,753 rows). The pipeline removes:

1. 247 rows dropped inside the source (5,971 → 5,724): 140 exact duplicates and 107
   normalised duplicates,
2. **5,118 matches against the UCI anchor** (5,724 → 606): 4,876 exact matches, 234
   near-duplicates at 0.90 cosine, and 8 audited exclusions (`dedup_exclusions.json`),
3. 7 rows within 0.90 cosine of an earlier source (606 → 599).

That is why only 599 of 5,971 rows survive. The kept rows are the ones that add information
UCI does not already contain. Do not quote 5,971 as the pool contribution.

## What was removed

See `data/processed/modern/data_audit.json` (`dedup.within`, `dedup.vs_uci`,
`dedup.cross_source`). Because the overlap is large, this source is a weak LOSO fold: 367 ham
and 232 positives remain.
