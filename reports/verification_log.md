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

## 2026-11-21 — G3: mmBERT fine-tune fits in 6 GB

Environment: RTX 3050 6 GB Laptop GPU (sm_86), CUDA 13.0 UMD 13.5,
torch 2.14.1+cu130, transformers 5.23.0, Python 3.14. The project venv shipped a
CPU-only torch; it was replaced with the cu130 wheel so the gate could run at all.
Both checkpoints were already in `models/`; nothing was downloaded for this test.

Command (UCI **train** rows only, 2000-row sample, seed 42, 1 epoch, embeddings
frozen, max_len 128, bf16 autocast, bs 16, lr 3e-5):

```bash
PYTHONPATH=src .venv/bin/python data/g3_smoke.py --model models/mmBERT-small
PYTHONPATH=src .venv/bin/python data/g3_smoke.py --model models/mmBERT-base \
  --out reports/results/g3_smoke_base.json
```

Numbers are quoted from `reports/results/g3_smoke.json` and
`reports/results/g3_smoke_base.json`:

| | mmBERT-small | mmBERT-base |
|---|---|---|
| parameters (trainable) | 140,642,306 (42,338,306) | 307,531,778 (110,923,778) |
| steps (125 = 2000/16) | 125 | 125 |
| median s/step | 0.079 | 0.174 |
| train wall clock, 1 epoch | 11.09 s | 23.45 s |
| tokens/s | 4,414 | 2,086 |
| **peak VRAM allocated** | **1.76 GiB** | **3.47 GiB** |
| peak VRAM reserved | 1.94 GiB | 3.62 GiB |
| CPU inference, 100 messages | 1.28 s (12.84 ms/msg) | 3.33 s (33.31 ms/msg) |

Both checkpoints are encoder-only: `AutoModelForSequenceClassification` reports
the classification head as newly initialised, which is expected for fine-tuning.

**Extrapolation.** At the measured throughput, at most 20,435 train rows (the
frozen pool train split, `data_audit.json`; each LOSO train set is smaller
because it excludes the held-out source) cost about 113 s/epoch for small and
240 s for base, so a 3-epoch fit is roughly 6 min and 12 min. Five LOSO fits
would be at most about 30 min (small) or 60 min (base) plus evaluation.
Extrapolation uses UCI-train throughput; pool text lengths were not measured, so
these are approximate, not bounds.

**G3 verdict: pass.** Base peaks at 3.47 GiB allocated (3.62 GiB reserved) at
bs 16 and max_len 128, leaving about 2.4 GiB of headroom on a 6 GB card, and it
is 2.2x slower than small. Recommendation: mmBERT-base for the headline model
and the final fit, mmBERT-small for the remaining LOSO folds (roadmap §7.10
compute plan). Drop to bs 8 with accum 2 only if longer `max_len` runs OOM.

---

## 2026-11-21 — label audit applied; pool hash superseded

`data/apply_audit.py` folded the reviewed sample back into the modern pool.
8 label correction(s) were applied before dedup, so the pool was rebuilt,
and the splits and LOSO folds were re-frozen at the same seed (42).

- previous dataset hash: `4015672722968db6` — **superseded**
- new dataset hash: `4b202c8f6ea707da`

| source | reviewed | disagree | rate | Wilson 95% |
|---|---|---|---|---|
| imc25 | 100 | 3 | 3.0% | 1.0%–8.5% |
| mishra | 100 | 1 | 1.0% | 0.2%–5.4% |
| moz | 100 | 6 | 6.0% | 2.8%–12.5% |
| smishtank | 100 | 0 | 0.0% | 0.0%–3.7% |

**Accepted above the gate: `moz` at 6.0% (Wilson 2.8%–12.5%).**

Reason: disagreement is one template that also appears in 232 of 2,009 ham rows, so it is label ambiguity, not a fixable rule

Its stored labels were left unchanged; the sampled corrections were not applied.
Row-level evidence is in `data/processed/modern/data_audit.json` under
`label_audit.accepted`.

Review rows from sources that are not in `POOL_SOURCES` were skipped: `exais`. Their overrides were filtered, not applied.
- `exais` excluded from the pool: class-definition mismatch: bulk/benign positives, not row noise (30% binary disagreement in the sample, Wilson 21.9%–39.6%; 27 of 30 were quotes and operator notices).

Only binary (ham vs malicious) disagreements change `y`; spam-vs-smish
disagreements change `y3` only. Any result computed on the previous hash
is stale and must be re-run.

## 2026-11-21 — gold holdout schema check

`gold_holdout.csv` was checked for schema only; its messages were not read.
133 rows, columns `id,text,label,language,scam_type,synthetic,notes`, no duplicate
ids, no duplicate texts, no blank labels.

- `label` is already canonical (`ham` 59, `smish` 60, `spam` 14), so the loader
  maps it straight to `y3` and derives `y = 0 if ham else 1`; no relabelling.
- `synthetic` is `1` on all 133 rows: the file is **fully synthetic** and was
  written by Claude on the user's behalf. Results on it are a synthetic stress
  test, not field evidence.
- Two new categories appear here that the pool does not contain: language
  `hinglish` (29 rows; the rest are `en`) and `scam_type` values `personal`,
  `promo`, `banking`, `delivery_update`, `bank_alert`, `utility`, `parcel`,
  `otp`, `job`, `appointment`.

The holdout is never used for tuning, thresholds or model selection, and no
misclassified gold message may be printed before E16.

---

## 2026-11-21 — Tier-2 classical runs: E5, E3, E4, E6 (+ gold, once)

> **Superseded.** The numbers below were computed on hash `4b202c8f6ea707da`
> (ExAIS excluded). The ExAIS ham-only rebuild changed the pool to
> `0bb9f4c85b6760d6`; see the rebuild entry at the end of this file.

**Read the base rate first.** Pool prevalence is 0.923, and IMC'25 is 86% of the
pool with only 3 ham rows. No pool-level accuracy or F1 is reported anywhere; a
pooled score would mostly measure IMC'25. Every table below is per source. FPR
is given for `mishra` and `moz` only; `imc25` and `smishtank` report recall only,
even though IMC'25 now holds 3 ham rows. The macro average covers the two
both-class folds only, so it has n = 2.

### E5 — formatting is nearly enough (pooled, per source, source identity)

Whole frozen pool, descriptive only: no model is selected and no threshold is set
here. Format-only features are twelve shape numbers (length, digit ratio, caps,
punctuation, URL/shortcode/phone flags); length-only is `n_chars` alone.

| probe | AUC |
|---|---|
| format-only, pooled | 0.978 |
| length-only, pooled | 0.792 |
| format-only, within `mishra` | 0.983 |
| format-only, within `moz` | 0.994 |
| length-only, within `mishra` | 0.902 |
| length-only, within `moz` | 0.843 |
| IMC'25 vs the rest, format-only | 0.964 |

Source identity from formatting alone is 0.933 accuracy against 0.865 chance.
The blunt statement: **IMC'25 rows are identifiable from shape alone at AUC
0.964**, and the separation survives inside a single source — 0.983 on Mishra and
0.994 on MOZ. So this is not only a cross-source artefact; within a source, the
positives are still much longer, more punctuated and more URL-shaped than the
ham. Any lexical model that reaches 0.97 on this pool is competing with a
formatting model that reaches 0.98.

### Headline selection (train only, written before any gold row was read)

Rule, in `reports/results/e3_selection.json`:
> Highest train-split out-of-fold macro PR-AUC over the two both-class sources
> (mishra, moz). Tie-break: lower train-OOF FPR at the OOF-fixed 1% threshold,
> then higher train-OOF recall. Test-split and LOSO numbers are not inputs.

| model | OOF macro PR-AUC | τ (OOF) | FPR @ τ | recall @ τ |
|---|---|---|---|---|
| **svm** | **0.9692** | 0.807 | 0.0093 | 0.931 |
| lr | 0.9565 | 0.877 | 0.0093 | 0.919 |
| cnb | 0.7387 | 1.000 | 0.0093 | 0.895 |
| mnb | 0.7387 | 1.000 | 0.0093 | 0.895 |
| dummy | 0.3287 | 1.923 | 0.0000 | 0.000 |

**Selected: `svm` (LinearSVC).** The pool test split was scored once with it, and
the gold pass used it. Dummy's 0.3287 is the base rate of the two both-class
sources (0.402 Mishra, 0.236 MOZ), not skill.

### E3 — pool train → pool test, one pass

| model | source | recall | FPR | PR-AUC |
|---|---|---|---|---|
| dummy | mishra | 0.000 | 0.000 | 0.405 |
| dummy | moz | 0.000 | 0.000 | 0.236 |
| dummy | imc25 / smishtank | 0.000 | — | — |
| dummy | macro (n=2) | 0.000 | 0.000 | 0.320 |
| mnb, cnb | mishra | 0.774 | 0.026 | 0.938 |
| mnb, cnb | moz | 0.000 | 0.000 | 0.547 |
| mnb, cnb | imc25 / smishtank | 0.919 / 0.808 | — | — |
| mnb, cnb | macro (n=2) | 0.387 | 0.013 | 0.743 |
| lr | mishra | 0.849 | 0.026 | 0.968 |
| lr | moz | 0.391 | 0.000 | 0.978 |
| lr | imc25 / smishtank | 0.941 / 0.832 | — | — |
| lr | macro (n=2) | 0.620 | 0.013 | 0.973 |
| svm | mishra | 0.887 | 0.013 | 0.965 |
| svm | moz | 0.655 | 0.000 | 0.989 |
| svm | imc25 / smishtank | 0.946 / 0.838 | — | — |
| svm | macro (n=2) | 0.771 | 0.006 | 0.977 |

"Always spam" on the pool test split scores recall 1.000, FPR 1.000, precision
0.923 and accuracy 0.923 — recorded in `e3_pool.json` only to make the base rate
visible, never as a result.

### E4 — leave one source out (same folds, no held-out rows in training)

| model | held-out | n_train | recall | FPR | PR-AUC |
|---|---|---|---|---|---|
| dummy | moz / mishra | | 0.000 | 0.000 | 0.234 / 0.402 |
| dummy | macro (n=2) | | 0.000 | 0.000 | 0.318 |
| mnb | moz | 16,611 | 0.000 | 0.000 | 0.313 |
| mnb | mishra | 17,682 | 1.000 | 0.997 | 0.730 |
| mnb | imc25 / smishtank | | 0.641 / 0.813 | — | — |
| cnb | moz | 16,611 | 0.000 | 0.001 | 0.313 |
| cnb | mishra | 17,682 | 1.000 | 0.997 | 0.739 |
| lr | moz | 16,611 | 0.139 | 0.066 | 0.407 |
| lr | mishra | 17,682 | 0.992 | 0.936 | 0.847 |
| svm | moz | 16,611 | 0.026 | 0.009 | 0.308 |
| svm | mishra | 17,682 | 0.989 | 0.819 | 0.852 |
| svm | imc25 / smishtank | | 0.383 / 0.756 | — | — |

The E4 macro (svm: recall 0.507, FPR 0.414) is an average of two opposite
failures and should not be quoted alone: the MOZ fold keeps the false-positive
rate and loses the recall (0.026), the Mishra fold keeps the recall and loses the
false-positive rate (0.819). Ranking transfers better than the threshold does on
Mishra (PR-AUC 0.852, ROC 0.897) and barely at all on MOZ (PR-AUC 0.308 against a
0.234 base rate).

### E6 — UCI-only vs pool-trained on the same folds

Paired deltas, per held-out source, at each arm's own OOF-fixed threshold:

| model | source | metric | uci_only | pool_unweighted | Δ | pool_source_weighted | Δ |
|---|---|---|---|---|---|---|---|
| svm | mishra | recall | 0.977 | 0.989 | +0.011 | 0.966 | −0.011 |
| svm | mishra | FPR | 0.031 | 0.796 | **+0.765** | 0.469 | **+0.439** |
| svm | mishra | PR-AUC | 0.991 | 0.852 | **−0.139** | 0.884 | **−0.107** |
| svm | moz | recall | 0.995 | 0.026 | **−0.969** | 0.092 | **−0.903** |
| svm | moz | FPR | 0.369 | 0.011 | −0.358 | 0.040 | −0.328 |
| svm | moz | PR-AUC | 0.345 | 0.308 | −0.037 | 0.362 | +0.017 |
| dummy | both | all | 0.000 | 0.000 | 0 | 0.000 | 0 |

Adding the modern pool as training data **makes both held-out sources worse on
PR-AUC** than UCI alone (Mishra 0.991 → 0.852/0.884; MOZ 0.345 → 0.308/0.362).
Source weighting helps ranking relative to unweighted on both sources, but not
enough to beat UCI-only. The pool arms also move the threshold badly: the
unweighted Mishra fold fires on 79.6% of Mishra ham.

### Gold — scored once, synthetic

133 rows, fully synthetic, written by Claude on the user's behalf. Scored after
the model and both thresholds were fixed from train-split evidence; never used
for selection, tuning or thresholding.

| model | arm | recall | FPR | PR-AUC |
|---|---|---|---|---|
| svm | uci_only | 0.689 | 0.559 | 0.599 |
| svm | pool_unweighted | 0.662 | 0.492 | 0.684 |
| dummy | either | 0.000 | 0.000 | 0.556 |

Dummy's 0.556 is the gold base rate (74 positive / 59 ham). The model is 0.04 to
0.13 PR-AUC above a constant classifier, and at the 1% FPR budget it flags
49%–56% of gold ham. By language (svm, pool arm): `en` recall 0.732 / FPR 0.562;
`hinglish` recall 0.444 / FPR 0.182. The model under-fires on hinglish rather
than over-firing. Worst `scam_type` recalls: `promo` 0.071, `wrong_number` 0.200,
`utility` 0.400, `banking` 0.769; `personal`, `bank_alert`, `otp`,
`delivery_update` are ham categories that fire at 0.500–1.000 FPR.

### Compute: what changed, and the measurements behind it

`cost_operating_point` called `metrics_at` for every candidate threshold, so each
call was O(thresholds × n) with an AP and a ROC-AUC per candidate. It is now an
O(n log n) sweep over the sorted scores. Verified against the old implementation
on 600 random score vectors with forced ties: same τ and same cost to 1e-9.

E6 vectorises each fold once and shares the matrix across all five models and
both pool arms. Measured on the Mishra fold with LR: vocabulary identical
(126,913 terms), test scores bitwise identical, OOF scores move (max |Δ| 3.63) and
τ shifts 0.5665 → 0.5487, test metrics unchanged. Wall time 33.8 s → 7.5 s.

Parallelism is over folds with processes, not threads over (model, fold). Two
threaded runs of the same E6 differed from each other in eight LinearSVC fields
(~1e-6 in PR-AUC, τ and cost τ), so liblinear's RNG state is shared across
threads. Two process runs are identical to each other and to the serial run. A
process per (model, fold) would refit the 7.5 s vectoriser five times for the same
wall clock, so the unit of work is a fold.

| run | wall time |
|---|---|
| E5 | 9 s |
| E3 (per-CV vectoriser) | 3 min 8 s |
| E4 (per-CV vectoriser) | 9 min 3 s |
| selection | 2 min 19 s |
| E6 paired, 6 workers | **46.6 s** (estimate was 25 min) |

E3 and E4 were not re-run, so they still use the stricter per-CV vectoriser. E6
uses the shared per-fold cache; the measured difference is above.

### Not run

- RF, LGBM and fastText (excluded on instruction).
- E7/E8 evasion suite, E9 held-out-family arms, E10 URL-masked arm, E16 error
  analysis, WSPR OOD, the synthetic Africa set.
- mmBERT LOSO (sequential on the GPU, no parallel fits on the 6 GB card).
- No second gold look, no rerun of E3/E4, no commits.

---

## 2026-11-21 — ExAIS ham-only rebuild, audited dedup exclusions, Tier-2 rerun

This entry supersedes the pool hash and every Tier-2 number above. Gold stayed
closed (`--skip-gold`); no gold row was read or scored.

### Why the pool was rebuilt

- **ExAIS is ham-only.** The label audit showed its "spam" class is
  bulk/unsolicited, not smishing (30% binary disagreement, Wilson 21.9%–39.6%;
  27 of 30 were quotes and operator notices). Every ExAIS spam row is dropped
  before dedup, its original ham labels are kept, and no ExAIS label is
  overridden. It is an FPR-only LOSO fold and never enters a both-class macro.
- **The near-dup rule is now corpus-independent.** The old char 3–4-gram TF-IDF
  fitted one pool-wide idf vocabulary, so adding or removing rows changed a
  score. A known Mishra-vs-UCI near-duplicate set moved below 0.90 when the rest
  of the pool was in the fit. The rule is now `use_idf=False`, `min_df=1`, fitted
  on exactly the two frames being compared, at 0.90 cosine.
- **The 8 rows the rule still misses are audited exclusions.** A pairwise audit
  found 36 Mishra rows that are near-duplicates of a UCI row. 28 clear 0.90; 8
  score 0.860498–0.898260. They are listed in
  `data/processed/modern/dedup_exclusions.json` and applied at the UCI-overlap
  stage. Every id must be present in the frame at that stage; the build raises
  on a stale id instead of skipping it. All 8 were present.
  - `20c93b03a1b52538`, `23fa6e94f1b3a001`, `268964bdf902d43a`,
    `30631ae09fd992ab`, `75961b28514f7681`, `79ab3811150d35e0`,
    `d5b861abdc6b2c90`, `f202a03d70916045`.
  - Reason recorded in `data_audit.json`: "audited pairwise near-dup of a UCI
    row; tf-only cosine 0.86-0.90, missed by the 0.90 rule".
  - **This list is audited, not exhaustive.** Other near-duplicates in the
    0.86–0.90 band may remain unaudited. Do not widen the list from another
    source without a new pairwise audit.

### Frozen pool (supersedes `4b202c8f6ea707da`)

Hash **`0bb9f4c85b6760d6`**. 30,447 rows: 25,963 positive / 4,484 ham,
prevalence **0.8527**.

| source | rows | ham | positive |
|---|---:|---:|---:|
| imc25 | 24,462 | 3 | 24,459 |
| mishra | 599 | 367 | 232 |
| moz | 2,330 | 1,784 | 546 |
| smishtank | 726 | 0 | 726 |
| exais | 2,330 | 2,330 | 0 |

| split | rows | positives | ham | prevalence |
|---|---:|---:|---:|---:|
| train | 19,487 | 16,616 | 2,871 | 0.8527 |
| cal | 4,870 | 4,153 | 717 | 0.8528 |
| test | 6,090 | 5,194 | 896 | 0.8529 |

Dedup stages: vs UCI 8 excluded + 4,878 exact + 234 near (30,616 out); then
cross-source 7 Mishra, 161 Smishtank and 1 ExAIS row dropped. LOSO folds are now
five: `mishra` and `moz` both-class, `imc25` and `smishtank` recall-only, `exais`
FPR-only. The both-class macro still has **n = 2**.

### Ham composition (counts only; no text read)

First matching rule wins: bank-OTP alert → transactional receipt → promo →
chatty. Rules are coarse and rule-based; they are not labels.

| source | chatty | transactional | bank-OTP | promo | total |
|---|---:|---:|---:|---:|---:|
| imc25 | 2 | 1 | 0 | 0 | 3 |
| mishra | 352 | 4 | 0 | 11 | 367 |
| moz | 1,753 | 15 | 5 | 11 | 1,784 |
| exais | 1,850 | 315 | 39 | 126 | 2,330 |
| **total** | **3,957** | **335** | **44** | **148** | **4,484** |

Strict bank-OTP (a code token *and* a bank/expiry token): 2, both ExAIS. Broad
bank-OTP: 44 (39 ExAIS, 5 MOZ). The pool now has real bank/OTP-style ham that it
did not have before; that is the point of keeping ExAIS ham.

### E5 — shortcuts

| probe | AUC |
|---|---:|
| format-only, pooled | 0.978 |
| length-only, pooled | 0.794 |
| format-only, within `mishra` | 0.984 |
| format-only, within `moz` | 0.994 |
| IMC'25 vs rest, format-only | 0.964 |
| ExAIS vs rest, format-only | 0.947 |

ExAIS ham: 2,330 rows, mean 110.7 chars, digit ratio 0.081, upper ratio 0.120,
14.0% with a double space. Format is still close to a source fingerprint, so no
pooled score is quoted.

### E3 — pool test, selected on train OOF only

Selection (train-OOF macro PR-AUC over `mishra` and `moz`): svm **0.9753**, lr
0.9642, cnb/mnb 0.8156, dummy 0.3354. **svm** remains the headline.

| svm, test | mode | n_pos | n_neg | recall | FPR | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| exais | fpr_only | 0 | 466 | — | 0.0322 | — |
| imc25 | recall_only | 4,893 | 0 | 0.9753 | — | — |
| mishra | both | 47 | 73 | 0.7021 | 0.0137 | 0.9727 |
| moz | both | 109 | 357 | 0.9174 | 0.0056 | 0.9818 |
| smishtank | recall_only | 145 | 0 | 0.9034 | — | — |
| **macro (n=2)** | | | | **0.8098** | **0.0097** | **0.9773** |

### E4 — LOSO, five folds

| svm, held out | mode | n_pos | n_neg | recall | FPR | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| moz | both | 546 | 1,784 | 0.3187 | 0.3111 | 0.2329 |
| mishra | both | 232 | 367 | 0.4871 | 0.0245 | 0.8616 |
| imc25 | recall_only | 24,459 | 3 | 0.1550 | — | — |
| smishtank | recall_only | 726 | 0 | 0.7259 | — | — |
| exais | fpr_only | 0 | 2,330 | — | 0.3811 | — |
| **macro (n=2)** | | | | **0.4029** | **0.1678** | **0.5473** |

The LOSO picture moved a lot against the pre-ExAIS run: MOZ recall rose 0.026 →
0.319 while Mishra recall fell 0.989 → 0.487, and macro PR-AUC fell 0.580 →
0.547. That is a pool-composition result, not a code change: every non-ExAIS fold
now trains with 2,330 ExAIS ham rows, and the Mishra/MOZ folds each train on a
different source mix. The 38% FPR on ExAIS ham is the cost of that shift and the
reason the fold is reported separately.

### E6 — UCI anchor vs pool arms (svm)

| arm | recall | FPR | PR-AUC |
|---|---:|---:|---:|
| uci_only | 0.9843 | 0.1953 | **0.6694** |
| pool_unweighted | 0.4010 | 0.1644 | 0.5473 |
| pool_source_weighted | 0.4438 | 0.1614 | 0.5826 |

Per source, the pool arms cut ExAIS FPR (0.376 → 0.274 weighted) and MOZ FPR
(0.369 → 0.301), but they lose 0.49–0.59 recall on the both-class sources and
0.08–0.13 macro PR-AUC. UCI-only is still the stronger ranking; the modern pool
is not a drop-in upgrade. Gold was not scored.

### Tests and checks

- `ruff check .` clean.
- Full suite: **123 passed**.
- New tests: score invariance under reference reordering/addition; the hardcoded
  36 (28 caught by 0.90 + 8 by the exclusion list, all 36 absent from the
  manifest); ExAIS ham-only filter and no label overrides; exclusion audit
  fields and stale-id raise; `fpr_only` LOSO mode and `write_results` by-mode
  keys; `mode_for("exais") == "fpr_only"` and `_macro` ignoring one-class rows.
- The hardcoded-36 test recomputes the pre-UCI frame from `data/raw` and skips
  when raw or processed data is absent.
- Gold remains closed; no gold row was read or scored.

## Next session

Full detail is in `docs/HANDOFF.md`. Text-free summary:

- **State:** pool hash `0bb9f4c85b6760d6`, 30,447 rows, prevalence 0.8527;
  `ruff` clean; 123 tests pass. Gold first pass is
  `reports/results/e6_gold.json` (`scored_once: true`, untouched).
- **Committed vs local-only:** code, tests, manifest, folds, audit, overrides,
  exclusions, results and docs are committed. Raw data, processed text
  (`train/cal/test.csv`), `review_sample.csv`, model weights and session exports
  stay git-ignored.
- **Open decision:** the user writes ~30 fresh gold messages, then one second
  gold pass. It must write `e6_gold_v2.json`; do not overwrite the first pass.
- **Next work:** mmBERT on the frozen pool (small for LOSO, base for the
  headline), then E7–E14.
- **Caveats:** ExAIS held-out FPR 0.381; ExAIS-vs-rest format AUC 0.947;
  UCI-only still beats the pool on macro PR-AUC (0.669 vs 0.547/0.583);
  near-dups in the 0.86–0.90 band may remain unaudited.
- **Gotchas:** E6 parallelism is process-per-fold (threaded liblinear is
  nondeterministic); E4 now takes ~20 min; the exclusion list raises on a stale
  id and must not be widened without a new audit; `score_gold` writes
  `e6_gold.json`, so point the second pass at `e6_gold_v2.json`.

