# DATA CARD — Smishtank Dataset I

| Field | Value |
|---|---|
| Source | `https://smishtank.com/analysisdataset.csv` (Dataset I, Timko & Rahman, CODASPY 2024, DOI `10.1145/3626232.3653282`) |
| Accessed | 2026-11-21 |
| Licence | **Unclear** — no machine-readable terms; cite the paper. Not redistributed |
| Rows (raw) | 1,062 |
| Labels | smishing only (`y3 = smish`) |
| Languages | English |
| Rows kept in the pool | 726 (read from `data/processed/modern/split_manifest.csv`) |
| Role | Extra positives; recall-only LOSO fold |

## What it is

1,062 validated smishing messages with URL and domain metadata (23 columns). 941 rows carry a
URL in the `Url` column. The public app API caps at 100 records and ignores pagination, so the
static CSV is the source of record.

## Decisions taken in the loader

- **`MainText` is the message body.** `Fulltext` prepends `Text Message\n<weekday>, <date>,
  <time>\n` and keeps OCR line breaks, which would hand the model a layout artefact. Using
  `MainText` removes that shortcut.
- `timeReceived` is carried as `date` (string, not parsed; no time split).
- `Message Categories` is carried as `scam_type`.
- The file is CRLF, extended ASCII, with embedded newlines inside quoted fields; read with
  `encoding_errors="replace"`.
- `Sender` is a brand, not a person. It is **not** used: brand names would be a direct shortcut.

## Known biases

- Positives only: no ham, so recall-only.
- 850 of 1,062 rows contain a URL (`has_url` on `MainText`, 80%), far above any other source. A
  model trained on the pool can learn "has a link" from this source alone; that is exactly what
  the E5 URL-shortcut probe and the E10 link-less slice must measure.
- 175 rows were dropped inside the source (1,062 → 887): 10 exact duplicates and 165 normalised
  duplicates. 161 more were within 0.90 cosine of an earlier source (mostly IMC'25), leaving 726.

## What was removed

Exact duplicates, 161 near-duplicates against earlier sources, and normalised duplicates. No
Smishtank row matched the UCI anchor. See `data/processed/modern/data_audit.json`.
