# Verification log

## 2026-11-21 — data acquisition

Environment: Linux, Python 3.14.10, pandas 2.3.3, network available.
Raw tree: `data/raw/` (git-ignored). Manifest: `data/README.md`.

### Commands run

```bash
# UCI
curl -L -o data/raw/uci_sms_spam_collection.zip \
  "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
unzip -q data/raw/uci_sms_spam_collection.zip -d data/raw/uci

# IMC'25
git clone --depth 1 https://github.com/reportsmishing/Smishing-Dataset-IMC25.git data/raw/imc25

# Mishra & Soni
curl -L -o data/raw/mishra_soni/Dataset_5971.zip \
  "https://data.mendeley.com/public-files/datasets/f45bkkt8pr/files/edb361de-918d-469f-9106-e84823830665/file_downloaded"
unzip -q data/raw/mishra_soni/Dataset_5971.zip -d data/raw/mishra_soni

# MOZ, India, Ngadou, combined, Africa (Hugging Face)
python - <<'PY'
from huggingface_hub import snapshot_download
for repo, dest in {
    "MOZNLP/MOZ-Smishing": "data/raw/moz",
    "CloveAI/india-spam-sms": "data/raw/hf/india-spam-sms",
    "Ngadou/Spam_SMS": "data/raw/hf/ngadou-spam-sms",
    "daviiiidcpp1/sms-spam-combined": "data/raw/hf/sms-spam-combined",
    "electricsheepafrica/africa-smishing-sms-phishing": "data/raw/hf/africa-smishing",
}.items():
    snapshot_download(repo, repo_type="dataset", local_dir=dest)
PY

# ExAIS and WSPR
git clone --depth 1 https://github.com/AbayomiAlli/SMS-Spam-Dataset.git data/raw/exais
git clone --depth 1 https://github.com/wspr-ncsu/sms-phishing.git data/raw/wspr

# Smishtank public API
curl "https://l0sry40g26.execute-api.us-west-1.amazonaws.com/prod/getsmish?type=smish" \
  -o data/raw/smishtank/smishtank_smishes.json
```

### Output of `python data/audit_sources.py`

```
UCI             5,574 rows; 4,827 ham / 747 spam; 403 exact duplicates (5,171 unique)
IMC'25         33,869 rows; text non-null 33,869; English 22,078; 157 absolute dates
Mishra & Soni   5,971 rows; 4,844 ham / 489 spam / 638 smishing
MOZ             2,561 rows; {'Legitimate': 2009, 'Smishing': 552}
ExAIS           5,240 rows; 2,890 ham / 2,350 spam
Smishtank         100 records via public API
WSPR           68,029 messages / 35,128 campaigns
India HF       20,010 rows; {0: 14059, 1: 5951}
Africa HF      10,000 rows; {1: 5000, 0: 5000}; synthetic={1: 10000}
Combined HF    55,404 train rows; {'ham': 34559, 'spam': 20845}
```

### Gate results

| Gate | Answer | Evidence |
|---|---|---|
| **G1** | IMC'25 has a real `text` column plus `translation`, `language`, `scam_type`, `lure_principles`. Dates are phone-UI strings (`Today`, `Yesterday`, localised) — only 157 absolute datetimes. **No time split.** | §1 above; `data/README.md` §2 |
| **G2** | LOSO-ready sources with both classes: MOZ (2,009/552), ExAIS (2,890/2,350), Mishra & Soni (4,844/1,127). Positives-only: IMC'25, Smishtank, WSPR. | `data/README.md` §2 |
| **G3** | Not started. `jhu-clsp/mmBERT-base` = 1,249 MB, `mmBERT-small` = 581 MB, both MIT. | HF API query |
| **G4** | No Hinglish data yet; gold holdout pending. | — |
| **G5** | Confirmed CC BY 4.0: UCI, IMC'25, Mishra & Soni. MIT: WSPR, India HF. ECL-2.0: combined HF. Unclear: MOZ card licence (model licence on a text dataset), ExAIS (none), Smishtank (none), Africa HF ("other"). | `data/README.md` §1 |

### Unresolved / blocked

- **NUS SMS Corpus:** original repo 404, SEACrowd mirror is a loader stub. UCI already includes
  3,375 NUS ham. Not blocking Tier 1/2.
- **Smishtank bulk:** public API caps at 100 records; pagination params (`page`, `offset`,
  `start`, `limit`, `count`) are ignored. 100 usable as recall-only OOD.
- **COVA / COVA-X:** papers found (arXiv 2604.11752, 2606.06879) but no dataset URL in either
  abstract or HTML. Tier 3 only; revisit.
- **`ealvaradob/phishing-dataset`:** 1.7 GB URL-phishing data, not SMS. Skipped.

### Checksum spot-check (SHA-256, first 16 hex)

```
UCI  SMSSpamCollection      7d039a24a6083ed9
IMC  final_dataset_output   1bbd1e9e82c3ea02
Mish Dataset_5971.csv       649844f1c62a6b27
MOZ  test.csv               814a11d9b05741c4
India HF parquet            8165c1d45a24d249
```
