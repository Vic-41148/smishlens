# DATA CARD — IMC'25 smishing

| Field | Value |
|---|---|
| Source | `reportsmishing/Smishing-Dataset-IMC25`, `dataset/final_dataset_output.csv` |
| Accessed | 2026-11-21 |
| Licence | CC BY 4.0 (repo README) |
| Rows (raw) | 33,869 |
| Labels | smishing only (`y = 1`, `y3 = smish`); no ham |
| Languages | English 22,078, Spanish 4,639, Dutch 1,945, French 1,163, German 810, + others |
| Rows kept in the pool | 24,462 |
| Role | Main modern positives; recall-only LOSO fold |

## What it is

Messages reported as smishing from public forums, with `scam_type`, `lure_principles`,
`language`, `url_shortener` and `named_entity` columns. Text is often transcribed from
screenshots, which is the main shortcut risk for this source (see roadmap §4.3).

## Decisions taken in the loader

- **`text` is the message column.** `translation` is present for only part of the corpus and is
  not used, so non-English rows stay in their original language and are tagged in `lang`.
- **`time` is not parsed.** It is a phone-UI string (`Today 4:53 PM`, `hoy, 19:28`, `今天 11:30`)
  and only 157 rows carry an absolute datetime. The time split was dropped (G1).
- `encoding_errors="replace"` — the file contains OCR junk in at least one row.
- `scam_type` is carried through; no label is derived from it.

## Known biases

- Positives only, so it contributes no ham. A model trained on this pool can learn
  "IMC'25 style" rather than smishing; the E5 shortcut probe exists to measure that.
- Screenshot transcription leaves layout artefacts (line breaks, OCR characters).
- 9,407 rows were dropped in within-source dedup before splitting (33,869 → 24,462): 9,106
  exact duplicates and 301 normalised duplicates, which is why the kept count is well below the
  raw count.

## What was removed

9,106 exact duplicates and 301 normalised duplicates within source
(`dedup.within.per_source.imc25`). No IMC'25 row matched the UCI anchor: `after_uci_overlap`
equals `after_within_dedup` in `data/processed/modern/data_audit.json`.
