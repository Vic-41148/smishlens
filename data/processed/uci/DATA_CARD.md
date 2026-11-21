# DATA_CARD — UCI SMS Spam Collection (anchor split)

| Field | Value |
|---|---|
| Source | UCI Machine Learning Repository, *SMS Spam Collection v.1* (Almeida, Gómez Hidalgo & Yamakami, 2011) |
| Local raw file | `data/raw/uci/SMSSpamCollection` (git-ignored) |
| Raw file SHA-256 | `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d` |
| Access date | 2026-11-21 |
| Licence | **CC BY 4.0** — redistribution permitted with attribution |
| Language | Predominantly English; a handful of non-English rows survive in the ham class |
| Task labels | binary: `spam` = 1, `ham` = 0 (`y3` keeps the original token) |
| Synthetic rows | none (`synthetic=False`) |
| Builder | `python data/build_uci_anchor.py` |
| Split seed | 42 (frozen) |
| Dataset hash (all rows) | `17edc73ba0692156` (`dataset_hash`, order-independent, truncated SHA-256) |

## Cleaning performed

| Step | Rows |
|---|---|
| Raw rows read | 5,574 |
| Exact duplicates removed (case/whitespace-normalised key) | 415 |
| Label conflicts removed | 0 (single source) |
| Rows kept | 5,159 |

The 415 duplicates are the 403 byte-identical repeats documented in `data/README.md`
plus 12 rows that differ only in case or whitespace and collide after the
`clean_dedup` key is applied. Dedup runs before the split, so no message can
appear in two splits.

No other filtering: the UCI corpus already has phone numbers and URLs replaced
with placeholders by its authors. `has_url` is computed on the raw text and is
therefore low (127 of 5,159 rows) — this is a property of the corpus, not of our
preprocessing.

## Frozen split (64 / 16 / 20, stratified by label)

| Split | Rows | Spam | Ham | Spam share | With URL | File |
|---|---|---|---|---|---|---|
| train | 3,301 | 411 | 2,890 | 12.5 % | 86 | `train.csv` |
| cal | 825 | 103 | 722 | 12.5 % | 20 | `cal.csv` |
| test | 1,033 | 128 | 905 | 12.4 % | 21 | `test.csv` |
| **total** | **5,159** | **642** | **4,517** | **12.4 %** | **127** | |

`cal` is used only for conformal quantiles and temperature scaling. `test` is
touched exactly once, by the final evaluation, and never for tuning.

## Known biases and limits

- **Origin composition.** Spam is dominated by the Grumble corpus (UK, 425
  messages) while ham is largely NUS SMS (Singapore students). Style differences
  between the two classes are partly *source* differences; this is the reason the
  modern pool exists and why LOSO is reported alongside in-pool numbers.
- **Era.** Messages are from roughly 2002–2011. Smishing link patterns of
  2024–2026 (shorteners, branded domains, OTP lures) are barely represented.
- **Class imbalance.** 12.4 % spam. Accuracy is meaningless here; every reported
  number is PR-AUC / F1 / MCC / recall at a fixed FPR.
- **URL scarcity.** Only 2.5 % of all rows contain a URL (19 % of spam, 0.3 % of
  ham), so URL-masking ablations have limited room to move the anchor metric.
- **Length.** Mean message length is short (SMS-scale); no long-form messages.
- **Anonymisation.** Placeholders (`xxx`, `0000`) are corpus artefacts. The
  normaliser maps them to stable tokens; do not treat them as real entities.

## Reproduce

```bash
python data/download_data.py          # rebuilds data/raw/ (git-ignored)
python data/build_uci_anchor.py       # rewrites data/processed/uci/
```

The CSVs in this directory are committed because the licence allows
redistribution. All other sources in `data/raw/` are not committed.
