# DATA CARD — ExAIS SMS Spam

| Field | Value |
|---|---|
| Source | GitHub `AbayomiAlli/SMS-Spam-Dataset` (20 `USER *.csv` files) |
| Accessed | 2026-11-21 |
| Licence | **Unclear** — no licence file in the repo. Research use only; not redistributed |
| Rows (raw) | 5,240 |
| Labels | 2,890 ham / 2,350 spam |
| Languages | English (Nigerian SMS, 2014–2015) |
| Rows kept in the pool | 3,648 (2,292 ham / 1,356 positive) |
| Role | Both-class LOSO source; the pool's second-largest ham contributor |

## What it is

Per-user exports of SMS from 20 users, with phone numbers masked. Older style (2014–2015),
Nigerian English, and a large ham share — useful precisely because it is not UCI-era UK/SG.

## Decisions taken in the loader

- **The files are ragged.** The column count changes row to row because message text contains
  commas and the header is absent. The loader therefore does **not** index by column position:
  it scans each row for the literal `HAM`/`SPAM` cell and re-joins every non-empty cell after it
  with `", "`.
  - Measured: the label cell is at index 6 in all 5,240 rows today, but the text spans 1–21
    cells, so position-based parsing would silently truncate most messages.
- Rows with no label token or empty text are dropped (0 today; the guard stays for re-downloads).
- `y3` is `ham` or `spam`; `y = 1` for spam.

## Known biases

- Timestamps (2014) are in the file but not used; no time split.
- Text is PII-masked by the authors, not by us. Some rows still carry placeholder numbers.
- The licence is unclear, so **processed text is never committed**; only the adapter and the
  text-free manifest row hashes are.

## What was removed

1,588 rows were dropped inside the source (5,240 → 3,652): 1,208 exact duplicates, 190 label
conflicts and 190 normalised duplicates. 3 rows were exact matches against the UCI anchor, 0
were near-duplicates of it, and 1 more was a near-duplicate of an earlier source. See
`data/processed/modern/data_audit.json`.
