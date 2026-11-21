# Data acquisition manifest

Accessed **2026-11-21**. Raw data lives under `data/raw/` and is **git-ignored**: it contains
real messages and PII, and some licences do not allow redistribution. Commit the download
script (`data/download_data.py`), not the payload.

Run `python data/download_data.py` to rebuild the tree.

---

## 1. Status by source

| # | Source | Local path | Status | Rows | Labels | Licence |
|---|---|---|---|---|---|---|
| 1 | UCI SMS Spam Collection v.1 | `raw/uci/SMSSpamCollection` | ✅ downloaded + verified | 5,574 (5,171 unique) | 4,827 ham / 747 spam | CC BY 4.0 |
| 2 | IMC'25 smishing | `raw/imc25/dataset/final_dataset_output.csv` | ✅ downloaded + verified | 33,869 | smishing only | CC BY 4.0 |
| 3 | Mishra & Soni (Mendeley) | `raw/mishra_soni/Dataset_5971.csv` | ✅ downloaded + verified | 5,971 | 4,844 ham / 489 spam / 638 smishing | CC BY 4.0 |
| 4 | MOZ-Smishing (HF) | `raw/moz/test.csv` | ✅ downloaded + verified | 2,561 | 2,009 legitimate / 552 smishing | card says `creativeml-openrail-m` (verify before redistributing) |
| 5 | ExAIS SMS Spam (GitHub) | `raw/exais/extracted/…/USER *.csv` | ✅ downloaded + parsed | 5,240 | 2,890 ham / 2,350 spam | no licence file in repo; research use, ask authors before redistributing |
| 6 | Smishtank (public API) | `raw/smishtank/smishtank_smishes.json` | ⚠️ partial | 100 | smishing only | site terms not machine-readable; verify before use |
| 7 | WSPR Lab SMS phishing (GitHub) | `raw/wspr/phishing_messages.csv` | ✅ downloaded | 68,029 msgs / 35,128 campaigns | phishing only | MIT (repo) |
| 8 | HF lead: `CloveAI/india-spam-sms` | `raw/hf/india-spam-sms/data/train-*.parquet` | ✅ downloaded | 20,010 | 14,059 ham / 5,951 spam | MIT |
| 9 | HF lead: `Ngadou/Spam_SMS` | `raw/hf/ngadou-spam-sms/spam_or_not_spam.csv` | ✅ downloaded | 5,574 | UCI mirror | CC (card) |
| 10 | HF lead: `daviiiidcpp1/sms-spam-combined` | `raw/hf/sms-spam-combined/data/*.parquet` | ✅ downloaded | 55,404 / 5,540 / 5,541 | ham/spam strings | ECL-2.0 |
| 11 | HF lead: `electricsheepafrica/africa-smishing-sms-phishing` | `raw/hf/africa-smishing/data/train-*.parquet` | ✅ downloaded | 10,000 | 5,000 / 5,000, **all `is_synthetic=1`** | "other" (verify) |
| 12 | NUS SMS Corpus | `raw/nus/` | ❌ not obtainable | — | — | — |
| 13 | COVA / COVA-X | — | ❌ no public link found | — | — | — |
| 14 | `ealvaradob/phishing-dataset` | — | ⏭️ skipped | 1.7 GB | URL phishing, not SMS | Apache-2.0 |

Sizes (raw tree, excluding `.git` and caches): see `du -sh data/raw/*` after download.

### Not obtained, and why

- **NUS SMS Corpus** — the historical GitHub repo (`WING-NUS/nus-sms-corpus`) is gone; the
  SEACrowd mirror is a loader stub with no data. Not a blocker: UCI already contains 3,375 NUS
  ham messages, so the ham signal is present. Revisit only if a live mirror appears.
- **COVA / COVA-X** — the two 2026 papers describe 3,201 and 10,985 synthetic conversations but
  publish no dataset URL in the abstract or HTML. Treat as Tier 3 / unavailable until a link is
  found. Do not substitute an unlabelled synthetic set.
- **Smishtank** — the site is a React app over a public API
  (`GET …/prod/getsmish?type=smish`) that returns **100 records per call** regardless of
  pagination parameters. The roadmap's ~1.1k figure cannot be confirmed from the public endpoint.
  Use the 100 as recall-only OOD, or contact the authors for the bulk release.
- **Super SMS** — still no confirmed public download (roadmap §4.2). Not used.

---

## 2. Gate findings

### G1 — IMC'25 columns and dates ✅ resolved

`dataset/final_dataset_output.csv`, 12 columns:

`time, sender_id, telephone_number_type, original_network_name, original_network_country,
text, translation, language, url_shortener, named_entity, scam_type, lure_principles`

- **Message text is a real column** (`text`). `translation` is present for ~12.7k/33.9k rows;
  `language` covers 22,078 English, 4,639 Spanish, 1,945 Dutch, 1,163 French, 810 German, …
- **Dates are not usable for a time split.** `time` is copied from phone UIs: `Today 4:53 PM`,
  `Yesterday 22:50`, `Tuesday 21 February 2023`, `hoy, 19:28`, `今天 11:30`, `21-2 12:44`.
  Only 157 rows parse as absolute datetimes; thousands are relative and localised. There is no
  capture timestamp column. **Decision: drop the TIME SPLIT.** Use LOSO instead (RQ1).
- Repo contains 33,869 labelled rows; the paper's 64.5k counts reports *and* attachments, so the
  CSV is the message-level subset. Fine for modelling, do not misquote 64.5k as row count.
- Text is often transcribed from screenshots → keep the source-shortcut probe (E5).

### G2 — Ham sources for LOSO ✅ resolved

Sources with **both classes** (valid LOSO folds):

| Source | Ham | Spam/smishing |
|---|---|---|
| MOZ-Smishing | 2,009 | 552 smishing |
| ExAIS | 2,890 | 2,350 spam |
| Mishra & Soni | 4,844 | 489 spam + 638 smishing |

Positives-only (recall-only OOD): IMC'25, Smishtank, WSPR. Synthetic only: the Africa HF set.
UCI is the anchor and contributes ham + spam.

### G3 — mmBERT memory

Not started. Weights identified:

| Repo | Size | Licence |
|---|---|---|
| `jhu-clsp/mmBERT-base` | 1,249 MB | MIT |
| `jhu-clsp/mmBERT-small` | 581 MB | MIT |

Do not download into the repo until the Day-10 smoke test; `artifacts/` and `data/raw/` are
git-ignored. Disk headroom at acquisition time was ~6 GB, so base + small + checkpoints fits but
is tight.

### G4 — Hinglish

No data yet; the gold holdout (Pool C) must be written before the first test run.

### G5 — Licences

Confirmed: UCI CC BY 4.0, IMC'25 CC BY 4.0, Mishra & Soni CC BY 4.0, WSPR MIT,
`india-spam-sms` MIT, `sms-spam-combined` ECL-2.0. Unclear: MOZ-Smishing (card says
`creativeml-openrail-m`, which is a model licence, not a text licence), ExAIS (no licence file),
Smishtank (no machine-readable terms), Africa HF set ("other"). Keep all unclear-licence data
out of any redistribution; do not commit raw text.

---

## 3. Verified counts (reproduce with `data/audit_sources.py`)

```
UCI            5,574 rows; 4,827 ham / 747 spam; 403 exact duplicate texts (5,171 unique)
IMC'25        33,869 rows; text non-null 33,869; English 22,078; 157 absolute dates
Mishra & Soni  5,971 rows; 4,844 ham / 489 spam / 638 smishing (case-normalised)
MOZ            2,561 rows; 2,009 Legitimate / 552 Smishing
ExAIS          5,240 rows; 2,890 ham / 2,350 spam (20 user files)
Smishtank        100 records via public API
WSPR          68,029 messages / 35,128 campaigns
India HF      20,010 rows; 14,059 ham / 5,951 spam
Africa HF     10,000 rows; 5,000/5,000; is_synthetic=1 for all
Combined HF   55,404 train (+5,540 val, 5,541 test); 34,559 ham / 20,845 spam
```

---

## 4. Known traps to enforce in the loaders

1. `Mishra & Soni` label casing is inconsistent (`spam`/`Spam`, `smishing`/`Smishing`). Normalise.
2. `Mishra & Soni` overlaps UCI heavily (~4.7k rows per the roadmap). Dedup exact + normalised
   before merging, and run the char-n-gram near-duplicate check.
3. `ExAIS` files are ragged CSVs: the column count changes per row, headers are data. Parse by
   locating the literal `SPAM`/`HAM` token, not by column index. The 20 files reproduce the
   paper's 5,240 / 2,350 / 2,890 exactly.
4. `IMC'25` has shifted rows (at least one row has OCR junk in `time`). Read with
   `encoding_errors="replace"`, never trust `time`.
5. `Africa HF` is fully synthetic. Tag `synthetic=True`; never in a headline test set.
6. `sms-spam-combined` inherits UCI and other sets. Treat as an aggregator; audit before training.
7. No raw personal SMS in git. Defang URLs/numbers in any published example.
