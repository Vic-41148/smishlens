# DATA CARD — ExAIS SMS Spam

| Field | Value |
|---|---|
| Source | GitHub `AbayomiAlli/SMS-Spam-Dataset` (20 `USER *.csv` files) |
| Accessed | 2026-11-21 |
| Licence | **Unclear** — no licence file in the repo. Research use only; not redistributed |
| Rows (raw) | 5,240 |
| Labels | 2,890 ham / 2,350 spam |
| Languages | English (Nigerian SMS, 2014–2015) |
| Rows kept in the pool | **2,330 — ham only** (0 positive). Every spam row is dropped before dedup |
| Role | **Ham-only negative source.** It trains as a negative source and is evaluated as an FPR-only LOSO fold; it never enters a both-class macro |

## What it is

Per-user exports of SMS from 20 users, with phone numbers masked. Older style (2014–2015),
Nigerian English, and a large ham share — useful precisely because it is not UCI-era UK/SG. The
surviving ham rows are the only bank/service-notification negatives the pool has.

## Decisions taken in the loader

- **The files are ragged.** The column count changes row to row because message text contains
  commas and the header is absent. The loader therefore does **not** index by column position:
  it scans each row for the literal `HAM`/`SPAM` cell and re-joins every non-empty cell after it
  with `", "`.
  - Measured: the label cell is at index 6 in all 5,240 rows today, but the text spans 1–21
    cells, so position-based parsing would silently truncate most messages.
- Rows with no label token or empty text are dropped (0 today; the guard stays for re-downloads).
- `y3` is `ham` or `spam`; `y = 1` for spam.

## Ham-only decision (2026-11-21)

The label audit found 30% binary disagreement on the spam class (Wilson 21.9%–39.6%). This is a
class-definition mismatch, not row noise: 27 of the 30 reviewed "spam" rows were benign quotes
and operator/bank notices. ExAIS is therefore never a positive source:

- every spam row (2,350) is dropped before dedup;
- no ExAIS label is overridden, and review corrections for ExAIS are filtered rather than applied;
- the surviving 2,330 ham rows keep their original labels.

## Known biases

- Timestamps (2014) are in the file but not used; no time split.
- Text is PII-masked by the authors, not by us. Some rows still carry placeholder numbers.
- About 6% of ExAIS ham is label noise: 3 of 50 reviewed ham rows were promos labelled ham. No
  correction was applied.
- The licence is unclear, so **processed text is never committed**; only the adapter and the
  text-free manifest row hashes are.

## What was removed

2,350 spam rows were dropped before dedup (5,240 → 2,890 ham). Inside the source, dedup removed
452 exact duplicates and 105 normalised duplicates (2,890 → 2,333). Against the UCI anchor, 2
rows were exact matches and 0 were near-duplicates. Cross-source near-dup removal dropped 1 more
row against an earlier source, leaving 2,330. See `data/processed/modern/data_audit.json`.
