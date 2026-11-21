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
Smishtank       1,062 smishing samples (Dataset I, positives only); 941 with a URL
WSPR           68,029 messages / 35,128 campaigns
India HF       20,010 rows; {0: 14059, 1: 5951}
Africa HF      10,000 rows; {1: 5000, 0: 5000}; synthetic={1: 10000}
Combined HF    55,404 train rows; {'ham': 34559, 'spam': 20845}
```

### Gate results

| Gate | Answer | Evidence |
|---|---|---|
| **G1** | IMC'25 has a real `text` column plus `translation`, `language`, `scam_type`, `lure_principles`. Dates are phone-UI strings (`Today`, `Yesterday`, localised) — only 157 absolute datetimes. **No time split.** | §1 above; `data/README.md` §2 |
| **G2** | LOSO-ready sources with both classes: MOZ (2,009/552), ExAIS (2,890/2,350), Mishra & Soni (4,844/1,127). Positives-only: IMC'25, Smishtank (1,062), WSPR. | `data/README.md` §2 |
| **G3** | Weights downloaded, **memory not measured**. `mmBERT-base` 1,249 MB, `mmBERT-small` 581 MB, both MIT. Smoke test still pending. | `models/`, `du -sh` |
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

---

## 2026-11-21 — second pass: full Smishtank, model weights, roadmap §7–§22

Disk before this pass: `/home` 36 GB free of 320 GB (90% used).

### Smishtank: 100-record API sample replaced by Dataset I

The app API caps at 100 records, but the dataset page links a static CSV. Fetched
`https://smishtank.com/analysisdataset.csv`.

```
1,062 rows, 23 columns, smishing only (no ham)
941 rows carry a URL
messageid, Fulltext, Sender, SenderType, timeReceived, MainText, Url, Subdomain, Domain, TLD,
RedirectedURL, Detected, Malicious, Phishing, Suspicious, Malware, Brand, URL Subcategory,
Message Categories, FullyQualifiedDomain, Domain Registrar, Domain Creation Date, Domain Last Update
```

Encoding note: the file is CRLF, non-ISO extended ASCII, and `Fulltext` holds embedded newlines
inside quoted fields. `pd.read_csv` needs `encoding_errors="replace"`. Paper: Timko & Rahman,
CODASPY 2024, DOI `10.1145/3626232.3653282`. The 100-record JSON is kept as a smoke sample only.

Consequence for the design: Smishtank stays **positives-only**, so it is a recall-only LOSO fold
and never a headline F1. It does add a URL-bearing slice (link-shortener and domain-age columns),
which the roadmap uses for the URL-shortcut probe.

### Sources that cannot be obtained

| Source | Status | Evidence |
|---|---|---|
| NUS SMS Corpus | **404** | `WING-NUS/nus-sms-corpus` appears in the org listing, but clone and codeload ZIP both 404; SEACrowd mirror is a loader stub. UCI already supplies 3,375 NUS ham. |
| COVA / COVA-X | **request only** | No dataset URL in either paper (arXiv 2604.11752, 2606.06879). `scamlingua.org` states non-commercial research access on request via `contact@scamlingua.org`. Not downloaded, not substituted. |

Both are recorded as blockers for Tier 3, not Tier 1/2.

### Model weights

All four are public third-party checkpoints. `models/` and `/var/lib/ollama` are git-ignored.

| Model | Local path | Bytes | SHA-256 (16) | Licence |
|---|---|---|---|---|
| `jhu-clsp/mmBERT-base` | `models/mmBERT-base/pytorch_model.bin` | 1,231,188,142 | `8ea64ec1ea4eb8fc` | MIT |
| `jhu-clsp/mmBERT-small` | `models/mmBERT-small/pytorch_model.bin` | 563,633,326 | `be7c9ce2af947fb5` | MIT |
| `telecomsxchange/OpenTextShield` | `models/OpenTextShield/model.safetensors` | 711,446,532 | `8e5f65efc7fecafc` | MIT |
| `qwen3:4b-instruct` | `/var/lib/ollama` (Ollama tag `0edcdef34593`) | 2.5 GB | — | Apache-2.0 |

Both mmBERT repos ship `pytorch_model.bin`, not `.safetensors`; only OpenTextShield uses
safetensors. Rebuild with `python data/download_models.py`.

**Correction to an intra-session mistake.** I first reported `~/.ollama` as 32 K and concluded the
`qwen3:4b-instruct` pull had failed, then killed it. Both were wrong: this host runs Ollama as a
system service storing blobs in `/var/lib/ollama`, and the pull was healthy at 596 MB / 2.5 GB.
Restarted, and it completed (`success`, digest verified). No data was corrupted by the kill; the
partial blob resumed. Do not use `~/.ollama` to judge pull progress on this machine.

### Roadmap

`ROADMAP.md` now holds the complete v4 document, §1–§22 (2,106 lines). Verified that the saved
§1–§6.5 text is byte-identical to the source file before appending §7–§22.

### Gate status after this pass

| Gate | Status | Note |
|---|---|---|
| G1 | ✅ resolved | IMC'25 has a real `text` column; timestamps unusable → no time split |
| G2 | ✅ resolved | 3 LOSO folds with both classes; 3 positives-only sources |
| G3 | ⚠️ **open** | weights present, peak VRAM still unmeasured — needs the Day-10 smoke test |
| G4 | ❌ open | no Hinglish gold holdout yet |
| G5 | ⚠️ partial | UCI/IMC'25/Mishra CC BY 4.0, WSPR/India HF/OpenTextShield/mmBERT MIT, combined HF ECL-2.0; MOZ, ExAIS, Smishtank, Africa HF unclear |

---

## 2026-11-21 — Tier-1 close-out (C1-C3)

### Disclosure: the test split was printed before the normaliser change

An earlier training run in this project (before defanged-URL masking was added to
`normalise.py`) printed UCI test-split metrics to the console. Those numbers were
**not** saved to `reports/results/` and are **superseded**. No decision was taken
from them: the threshold is frozen from out-of-fold train scores, the conformal
quantiles come from `cal`, and neither reads test. The single authoritative
`reports/results/test_metrics.json` was written after the retrain below, once.

### What was rebuilt

| Step | Command | Result |
|---|---|---|
| Anchor split materialised | `python data/build_uci_anchor.py` | 5,159 rows after dedup (415 duplicates dropped, 0 label conflicts); 3,301 / 825 / 1,033 |
| Tier-1 experiments | `python -m smishlens.experiments` | E1/E2/E3 on train only, 5-fold CV, PR-AUC primary |
| Final retrain | `python -m smishlens.train` | canonical `artifacts/` (was `artifacts/uci`), tau from OOF, Mondrian on `cal` |
| One-shot test evaluation | `python -m smishlens.final_eval` | wrote `reports/results/test_metrics.json` |

### Reading the experiments

- On UCI, E1 arms tie within one fold-std. That is expected: the anchor has almost
  no Unicode/defanging obfuscation, so the normaliser has nothing to fix. The
  normaliser's value must be shown on the modern pool and the evasion suite, not here.
- E2: word + char TF-IDF is the best feature set (PR-AUC 0.9848 ± 0.0044); adding
  meta features is neutral-to-slightly-negative on this anchor.
- E3: LinearSVC wins (PR-AUC 0.9845 ± 0.0064), ahead of LR/CNB/MNB/RF, and is the
  classifier already used by the production pipeline. Champion confirmed, not changed.

### Canonical artifact directory

`artifacts/` is the only artifact root. `smishlens.train` now defaults to it;
`serve/api.py` and `app/app.py` already read `ARTIFACT_DIR` with default
`artifacts`. `artifacts/uci` was removed so two models cannot be confused.

### Known limitation

The anchor test split has 128 spam rows, so one evaluation is a small sample: the
bootstrap 95% interval for recall is 0.946-1.000 and for precision 0.935-0.992
(`reports/results/test_metrics.json`). Quote the intervals, not the point estimates
alone. Conformal coverage on test is 0.948 (ham) / 0.977 (spam) at alpha = 0.05,
consistent with the 95% target on this in-pool split. Coverage under a held-out
source is the separate question RQ5 studies; nothing here establishes it.

---

## 2026-11-21 — determinism audit: liblinear was unseeded

The v1.0-mvp test numbers in `reports/results/test_metrics.json` came from a
`LinearSVC(dual="auto")` fit with **no `random_state`**. With 3,301 samples and
27,429 features, `dual="auto"` selects liblinear's dual coordinate-descent solver,
which permutes coordinates randomly on every fit. The score therefore moves by
roughly 1e-6 between retrains, and so does `tau`, which is a quantile of the
out-of-fold scores. This was found during the C5 fresh-clone check: the clone
retrained `tau = 0.15136295402385722` against the reported `0.15136336820370772`.

### What changed

`random_state=42` (or `SEED`) added to every estimator that had one unset:

| File | Estimator |
|---|---|
| `src/smishlens/features.py` | `LinearSVC` inside `CalibratedClassifierCV` (the production model) |
| `src/smishlens/models.py` | `LinearSVC` and `LogisticRegression` in `GRIDS` |
| `src/smishlens/experiments.py` | `LR` and the E3 `LinearSVC` |
| `src/smishlens/export_json.py` | `LinearSVC` in the JSON student |

Already seeded and left alone: every `StratifiedKFold(..., shuffle=True)` (42/`SEED`),
`train_test_split` in `data.py` and `train.py`, `RandomForestClassifier` and
`HistGradientBoostingClassifier`, `np.random.default_rng` in `conformal.py` and
`evaluate.py`, `random.Random(seed)` in `attacks.py`. `CalibratedClassifierCV(cv=5)`
takes the int form, which sklearn resolves to a **non-shuffled** stratified split, so
it is deterministic without a seed. `train_mmbert.py` seeds the global NumPy RNG; it
is not on the Tier 1 path and will be revisited at the G3 smoke test.

### What was verified — train and calibration only, test not scored

- Two consecutive seeded retrains produced **bit-identical** `tau`
  (`0.15136301338984764`), identical `q_hat`, and identical calibration-split
  probabilities (SHA-256 `447697e8251c02db6aee3eeec90a666dbf16a426375bb3de47f1c9fe4700bf94`
  over the float64 probability vector).
- E1/E2/E3 were regenerated on the train split and are **byte-identical** to the
  committed `reports/results/tier1_*.csv`. No number moved, at 4 dp or below.

### What was not done

`final_eval.py` was **not** re-run and `reports/results/test_metrics.json` was **not**
touched. The seeded `tau` differs from the reported one by 3.55e-7 absolute
(2.34e-6 relative); the reported test metrics therefore stand as evaluated under the
unseeded fit. No claim is made here about how many test decisions, if any, that
difference would flip — that was not measured, and measuring it would mean scoring
the test split again.
