# DATA CARD — MOZ-Smishing

| Field | Value |
|---|---|
| Source | Hugging Face `MOZNLP/MOZ-Smishing`, `test.csv` |
| Accessed | 2026-11-21 |
| Licence | **Unclear** — the dataset card states `creativeml-openrail-m`, which is a *model* licence, not a text licence. Not redistributed |
| Rows (raw) | 2,561 |
| Labels | 2,009 Legitimate / 552 Smishing |
| Languages | Portuguese (Mozambique) |
| Rows kept in the pool | 2,330 (1,784 ham / 546 positive) |
| Role | Both-class LOSO source; the only Portuguese fold |

## What it is

Crowd-sourced mobile-money smishing and legitimate messages from Mozambique. `id, source, text,
label`. The `source` column records the collection channel, not a dataset name.

## Decisions taken in the loader

- `label` is lower-cased and mapped: `Legitimate → ham`, `Smishing → smish`.
- `lang` is set to `pt` for every row (single-language corpus, declared on the card).
- `source` (collection channel) is not used as a feature — it would be a direct shortcut.

## Known biases

- Mobile-money framing dominates the positives, so a model can learn "mpesa/bank vocabulary"
  rather than smishing structure.
- Only 33 rows contain a URL in the raw text (`has_url`), so this source contributes almost
  nothing to the URL/link-less contrast (E10).
- Licence unclear: processed text is not committed.

## What was removed

231 rows were dropped inside the source (2,561 → 2,330): 3 exact duplicates, 8 label conflicts
and 220 normalised duplicates. No MOZ row matched the UCI anchor and no near-duplicate was found
against earlier sources. See `data/processed/modern/data_audit.json`.
