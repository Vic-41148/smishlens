# Verification log

> **Note (2026-11-22).** This log is a record of what was run, not a live
> index. The public history was rewritten on this date to purge the
> text-bearing splits (`data/gold_holdout*.csv`,
> `data/processed/uci/{train,cal,test}.csv`). Short commit ids written before
> this date refer to pre-rewrite commits and no longer resolve; the two
> gold-freeze commits were dropped entirely because the split was their only
> change. Nothing else in the log was edited.


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

---

## 2026-11-21 — G3 mmBERT smoke test (6 GB card)

First GPU run of `smishlens.train_mmbert`. Local checkpoints
`models/mmBERT-base` and `models/mmBERT-small`; `HF_HUB_OFFLINE=1`, so no
network access. Subset: 2,000 stratified train rows (1,705 positive / 295 ham),
one epoch, `max_len=128`, `bs=16`, embeddings frozen, no gradient checkpointing.

| model | peak VRAM | wall clock | s/step | tokens/s |
|---|---:|---:|---:|---:|
| mmBERT-base | **3.39 GiB** | 34.6 s | 0.277 | 2,435 |
| mmBERT-small | **1.73 GiB** | 16.1 s | 0.129 | 5,225 |

**G3 verdict: pass.** Base fits with ~2.4 GiB headroom on the 6 GB card.
Extrapolation to the frozen train split (19,487 rows, 3 epochs, 3,654 steps):
about **17 min per base fit** and **8 min per small fit**, before evaluation.
That makes base-E3 plus small-E4 (five folds) feasible in about one hour.

Decision: **base for the E3 headline, small for the five LOSO folds.** The
training loop emits a benign scheduler-before-optimizer warning on the first
GradScaler-skipped step; it is recorded here rather than changed mid-run.

### E3 — mmBERT-base headline

Trained on pool `train` (19,487 rows), calibrated on `cal` (4,870; `cal_a` fits
temperature and tau, `cal_b` fits conformal), and the test split was scored once.
Three epochs, `max_len=128`, `bs=16`, embeddings frozen, seed 42. Peak VRAM
3.39 GiB; fitted `T=2.543`, `tau=0.9610`. Checkpoint:
`artifacts/mmbert/e3_base/mmbert` (git-ignored), metadata beside it.

| mmBERT-base, test | mode | n_pos | n_neg | recall | FPR | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| exais | fpr_only | 0 | 466 | — | 0.0129 | — |
| imc25 | recall_only | 4,893 | 0 | 0.9826 | — | — |
| mishra | both | 47 | 73 | 0.7872 | 0.0000 | 0.9864 |
| moz | both | 109 | 357 | 0.9908 | 0.0112 | 0.9765 |
| smishtank | recall_only | 145 | 0 | 0.9655 | — | — |
| **macro (n=2)** | | | | **0.8890** | **0.0056** | **0.9814** |

Against the classical svm headline on the same test split (macro 0.8098 /
0.0097 / 0.9773), mmBERT-base gains 0.079 macro recall and 0.004 macro PR-AUC
and cuts macro FPR by 0.004. Per source it gains on every recall-only fold and
on Mishra recall, while MOZ PR-AUC slips slightly (0.9818 → 0.9765). ExAIS FPR
is 0.0129 on test although the threshold was frozen at ≤1% on `cal_a`; that is
the expected small-sample drift, not a tuning choice.

### mmBERT provenance, thresholds and test hygiene (text only)

Written while the E4 small run was still in progress. No new run was started for
this note and no gold row was read.

**Threshold provenance.**

- **E3 base:** temperature `T` and `tau` come from `cal_a`, the first half of the
  frozen `cal` split (stratified, seed 42). Conformal `q_hat` comes from
  `cal_b`. No test row is used to fit `T`, `tau` or `q_hat`.
- **E4 small:** in each fold, `T` and `tau` come from `cal_a` of the `cal` rows
  that remain after removing the held-out source; `q_hat` comes from `cal_b` of
  those rows. The held-out source is never in train or cal. No held-out row is
  used to fit `T`, `tau` or `q_hat`.
- Classical models keep their train-OOF threshold. mmBERT has no OOF pass, so
  `cal_a` is the documented substitute. This is a deliberate protocol
  deviation, not a test-derived threshold.

**Test scoring.**

- **E3 base:** the frozen pool test split was scored once, after training and
  calibration were frozen. The reported numbers come from that one scoring.
- **E4 small:** each held-out source is scored once per fold, at its frozen split
  labels. No fold is re-scored or tuned after seeing its numbers.
- **Plumbing check:** before the final runs, a throwaway 600-row subset (40 rows
  per split/source, 1 epoch, `max_len=64`, small model) was scored to validate
  the new code path. It was not a final model, was not used for any threshold or
  selection, and its checkpoint was deleted. It is not a reported result.

**Base/small assignment.**

- **There is no train/cal performance rule that picked base for E3 and small for
  E4.** The assignment is role/compute-based and was fixed before any test or
  LOSO number: base is the headline/final model; small is the LOSO workhorse
  because G3 measured 1.73 GiB and about twice the throughput (0.129 vs 0.277
  s/step). The roadmap's §7.10 compute plan says small on every fold and base for
  the headline/final model. We did not compare base vs small on test or LOSO and
  then choose; no such comparison exists.
- If a performance-based rule is wanted later, it must be defined on train/cal
  evidence before the next test/LOSO look.

**Checkpoints and reproducibility.**

- **E3 base:** `artifacts/mmbert/e3_base/mmbert` (git-ignored). Manifest:
  `artifacts/mmbert/e3_base/MANIFEST.sha256`. Model SHA256:
  `8edbc7d9da486237d6ffdf27456e7845a5367b9daef6bc55930ab1422addcc3c`.
  Metadata: `model_name=mmbert_base_temp_scaled`, seed 42, epochs 3,
  `max_len=128`, `bs=16`, embeddings frozen, no gradient checkpointing,
  train 19,487, cal 4,870, `T=2.543`, `tau=0.9610`.
- **E4 small:** `artifacts/mmbert/e4_small/heldout_<source>/mmbert` per fold,
  each with `MANIFEST.sha256` and `metadata.json`. Same seed 42, 3 epochs,
  `max_len=128`, `bs=16`, frozen embeddings, no gradient checkpointing. All five
  folds finished; per-fold hashes are in the E4 section below.
- Source checkpoints: `models/mmBERT-base` and `models/mmBERT-small`, loaded
  offline with `HF_HUB_OFFLINE=1`.

**Gold.**

- mmBERT never loaded or scored gold. `e6_gold.json` remains the untouched first
  pass (`scored_once: true`); no `e6_gold_v2.json` exists.
- Gold stays closed until the user writes the ~30 fresh messages. Then exactly
  one pass, written to `e6_gold_v2.json`; the first pass is not overwritten.

### E4 — five-fold LOSO, mmBERT-small vs classical svm

Same five held-out folds for both models. Each fold trains on pool `train` rows
without the held-out source, calibrates on `cal` rows without that source, and
scores the held-out source once. mmBERT-small used `cal_a` for `T`/`tau` and
`cal_b` for conformal; the classical svm keeps its train-OOF threshold. Macro is
over the two both-class folds only (`mishra`, `moz`, n=2).

| held out | mode | n_pos | n_neg | svm recall | small recall | svm FPR | small FPR | svm PR-AUC | small PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| imc25 | recall_only | 24,459 | 3 | 0.1550 | 0.4286 | — | — | — | — |
| smishtank | recall_only | 726 | 0 | 0.7259 | 0.6460 | — | — | — | — |
| mishra | both | 232 | 367 | 0.4871 | 0.4957 | 0.0245 | 0.0545 | 0.8616 | 0.7748 |
| moz | both | 546 | 1,784 | 0.3187 | 0.7125 | 0.3111 | 0.3812 | 0.2329 | 0.3654 |
| exais | fpr_only | 0 | 2,330 | — | — | 0.3811 | 0.2948 | — | — |
| **macro (n=2)** | | | | **0.4029** | **0.6041** | **0.1678** | **0.2178** | **0.5473** | **0.5701** |

Reading:

- mmBERT-small raises macro recall by 0.201 and macro PR-AUC by 0.023, and pays
  0.050 more macro FPR. The gain is concentrated in `moz` (recall 0.319 → 0.713,
  PR-AUC 0.233 → 0.365) and `imc25` (0.155 → 0.429).
- On `mishra`, recall is flat (0.487 → 0.496) and PR-AUC drops 0.087. On
  `smishtank`, recall drops 0.080.
- ExAIS FPR improves (0.381 → 0.295) but is still far above the 1% calibration
  budget on both models. That is the LOSO shift the fold is meant to expose.
- Both models are weak under source shift. mmBERT does not fix LOSO; it moves
  the failure around. This is the honest E4 result, not a headline.

Per-fold fit settings (all seed 42, 3 epochs, `max_len=128`, `bs=16`, embeddings
frozen):

| held out | train rows | cal rows | T | tau | model.safetensors SHA256 |
|---|---:|---:|---:|---:|---|
| exais | 17,996 | 4,497 | 1.796592 | 0.988252 | `276bfda1b977bb3f1e91168fe577211dcfae0540e1f92744486870a892d13867` |
| imc25 | 3,830 | 958 | 4.060639 | 0.771195 | `2b08b83686d0167f0661cf3d66a291a69a19cc5c9767f598391d9a80e2b5619f` |
| mishra | 19,104 | 4,774 | 2.039492 | 0.981982 | `e329118db1101f1a9e8b93dc5e93cfce46d816be98b6308c8f5cf4aeb003725a` |
| moz | 17,996 | 4,497 | 2.529097 | 0.983906 | `c294086a9a4680c9ccb454f7774a5fbf38dcd7c1308b7b4dee45afa55bb4b7ce` |
| smishtank | 19,022 | 4,754 | 2.359537 | 0.993047 | `dcd714a29e9777de78f979df8db64dbeeb122bc3f740c6b96b8a3aed635bb882` |

Manifests: `artifacts/mmbert/e4_small/heldout_<source>/MANIFEST.sha256`
(checkpoints git-ignored). Result:
`reports/results/e4_loso_mmbert_small.json`. Classical comparison:
`reports/results/e4_loso.json` (`svm` rows).

Test hygiene: no fold was re-scored or re-tuned after its numbers were seen; no
gold row was read. Gold stays closed.






---

## 2026-11-21 — Phase 1 close-out: gold v2, E12-E14, RQ1 gap

Text-free entry. No gold message, id or note appears here or in any output
file. Gold was not used for tuning or selection at any point.

### Commit (c): gold v2 frozen before any model saw it

`99b21ce` — signed, `git log --show-signature -1` reports a good signature.
`data/gold_holdout_v2.csv`, 58 rows, ham 35 / smish 19 / spam 4, en 44 /
hinglish 14, `synthetic=1` on every row, no duplicate ids. Written by Claude
on the user's behalf and frozen here. `ruff` clean and 126 tests passed before
the commit. The temporary GPG pinentry config was restored afterwards and the
agent restarted.

### Gold pass 2 — `reports/results/e6_gold_v2.json`

One pass, one write. v1 (133 rows) and v2 (58 rows) are separate blocks.
`e6_gold.json` was not touched.

**Provenance, stated plainly:** v1 was first scored before the ExAIS ham-only
rebuild, so it is no longer pristine for that decision; this pass is a repeat
for v1. v2 was frozen before any model scored it and is pristine.

| gold | model | recall | FPR | PR-AUC |
|---|---|---:|---:|---:|
| v1 (133) | svm (pool) | 0.7162 | 0.4915 | 0.7479 |
| v1 (133) | mmBERT-base E3 | 0.8784 | 0.5424 | 0.8767 |
| v1 (133) | Dummy | 0.0000 | 0.0000 | 0.5564 |
| v2 (58) | svm (pool) | 0.6957 | 0.5429 | 0.6850 |
| v2 (58) | mmBERT-base E3 | 1.0000 | 0.4857 | 0.8483 |
| v2 (58) | Dummy | 0.0000 | 0.0000 | 0.3966 |

By language, v2 (aggregate only):

| model | language | n | recall | FPR |
|---|---|---:|---:|---:|
| svm | en | 44 | 0.7059 | 0.6667 |
| svm | hinglish | 14 | 0.6667 | 0.1250 |
| mmBERT-base | en | 44 | 1.0000 | 0.5556 |
| mmBERT-base | hinglish | 14 | 1.0000 | 0.2500 |

The by-`scam_type` block is in the JSON. The aggregate pattern is the story:
on v2, mmBERT-base flags 7/7 ham `bank_alert` rows and 2/2 ham `utility` rows;
`personal` ham is 0/8 flagged. svm shows the same direction less sharply. No
row is reproduced here.

Reading:

- The frozen 1% pool threshold does not transfer to gold. Pool test FPR was
  0.0056 (base) and 0.0097 (svm); gold FPR is ~0.49-0.54 for both. Recall
  stays high, and PR-AUC stays 0.85 (base) / 0.69 (svm) on v2, so the ranking
  is far better than the thresholded decision.
- The failure is concentrated in legitimate bank/service messages. A model
  trained to read "bank + urgency + link" as spam blocks real bank alerts.
  This is a threshold-transfer failure, not a ranking failure, and it is the
  most important gold result so far.
- Gold is synthetic and Claude-written. Treat the absolute FPR as a warning
  about style shift as well as about the model; do not quote it as a field
  false-positive rate.
- Hinglish FPR is lower than en on both models. With 8 Hinglish ham rows in
  v2 this is not evidence; it is a note for the Phase 2 Hinglish slice.

### E13 — calibration on `cal_b` only

`reports/results/e13_calibration.json`, `reports/figures/e13_reliability.png`.

| model | n | ECE (10 bins) | Brier | mean confidence | base rate |
|---|---:|---:|---:|---:|---:|
| svm (pool) | 2,435 | 0.0064 | 0.0087 | 0.8554 | 0.8526 |
| mmBERT-base | 2,435 | 0.0054 | 0.0046 | 0.8480 | 0.8526 |

Method: `cal_a` fits the map, `cal_b` is evaluated. The pool svm is a raw
`LinearSVC` decision function, so Platt (`sigmoid(a*score+b)`) was fitted on
`cal_a` decision scores and applied to `cal_b`; no probability was invented
before `cal_a`. mmBERT uses its stored temperature `T=2.542974`, already fitted
on `cal_a`. Nothing was fitted on `cal_b`.

Caveat: `cal_b` is also the half the E3 conformal `q_hat` was fitted on, so it
is held out for temperature/Platt but not unused. It is 85% spam and both
models are bimodal, so ECE is dominated by the two extreme bins. The figure
marks sparse bins but does not connect them; low ECE here does not certify the
mid-confidence range, which has very few rows.

### E12 — PPV at 1%, 5%, 10% prevalence

`reports/results/e12_ppv.json`. Arithmetic only, from frozen E3 (in-pool) and
E4 (LOSO) recall/FPR. No test row was re-scored.

In-pool, E3 test thresholds:

| model | source | recall | FPR | PPV@1% | PPV@5% | PPV@10% |
|---|---|---:|---:|---:|---:|---:|
| svm | mishra | 0.7021 | 0.0137 | 0.3411 | 0.7296 | 0.8506 |
| svm | moz | 0.9174 | 0.0056 | 0.6232 | 0.8960 | 0.9479 |
| svm | macro | 0.8098 | 0.0097 | 0.4822 | 0.8128 | 0.8993 |
| mmBERT-base | mishra | 0.7872 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| mmBERT-base | moz | 0.9908 | 0.0112 | 0.4718 | 0.8231 | 0.9076 |
| mmBERT-base | macro | 0.8890 | 0.0056 | 0.7359 | 0.9116 | 0.9538 |

Under LOSO, E4 frozen fold thresholds:

| model | source | recall | FPR | PPV@1% | PPV@5% | PPV@10% |
|---|---|---:|---:|---:|---:|---:|
| svm | mishra | 0.4871 | 0.0245 | 0.1671 | 0.5111 | 0.6882 |
| svm | moz | 0.3187 | 0.3111 | 0.0102 | 0.0512 | 0.1022 |
| svm | macro | 0.4029 | 0.1678 | 0.0887 | 0.2811 | 0.3952 |
| mmBERT-small | mishra | 0.4957 | 0.0545 | 0.0841 | 0.3237 | 0.5026 |
| mmBERT-small | moz | 0.7125 | 0.3812 | 0.0185 | 0.0896 | 0.1720 |
| mmBERT-small | macro | 0.6041 | 0.2178 | 0.0513 | 0.2067 | 0.3373 |

**Label:** E4 LOSO exists for classical svm and mmBERT-**small**, not for
mmBERT-base. The LOSO rows must not be read as base-model results. The base
model was never LOSO-tested.

How PPV collapses: at 1% prevalence, in-pool macro PPV is 0.48 (svm) and 0.74
(base); under LOSO it is 0.09 (svm) and 0.05 (small). On `moz` under LOSO, svm
PPV@1% is 0.010, i.e. roughly one true alert per hundred. A 31% FPR makes a
low-prevalence alert stream almost all false alarms, regardless of recall.

### E14 — conformal coverage vs the 0.95 target

`reports/results/e14_conformal.json`. Nominal coverage 0.95, `alpha=0.05`,
class-conditional (Mondrian) sets.

In-pool (cal_a fits `q_hat`, cal_b is evaluated):

| model | ham coverage | spam coverage | overall | abstain |
|---|---:|---:|---:|---:|
| svm (pool) | 0.9359 | 0.9624 | 0.9585 | 0.0374 |
| mmBERT-base | 0.9805 | 0.9552 | 0.9589 | 0.0386 |

20 random 50/50 resplits of the calibration split (mean, min-max):

| model | cov_ham | cov_spam | abstain |
|---|---|---|---|
| svm | 0.9536 (0.9213-0.9769) | 0.9481 (0.9326-0.9639) | 0.0466 (0.0382-0.0600) |
| mmBERT-base | 0.9516 (0.9160-0.9827) | 0.9497 (0.9372-0.9630) | 0.0479 (0.0357-0.0616) |

LOSO (mmBERT-small frozen fold checkpoints, no refit; `q_hat` from each fold's
`metadata.json`):

| held out | ham coverage | spam coverage | overall | abstain |
|---|---:|---:|---:|---:|
| mishra | 0.5368 | 0.4310 | 0.4958 | 0.4157 |
| moz | 0.1715 | 0.7271 | 0.3017 | 0.3901 |

The in-pool guarantee holds at the nominal level. Under source shift it does
not: overall coverage falls to 0.50 on mishra and 0.30 on moz, while the
abstain rate rises to ~39-42%. Conformal abstention does not repair the shift;
it converts some errors into abstentions but still leaves most predictions
uncovered. Resplits are not run for LOSO because `q_hat` is frozen per fold;
that is deliberate, not missing work.

**Repeat disclosure:** E14 LOSO re-scores the frozen E4 checkpoints
(`artifacts/mmbert/e4_small/heldout_<source>`) with no refit. It is a
deterministic repeat of a model already scored once, not a new look at a
retrained model. The per-fold scores were not saved in E4, so this is the only
way to get coverage without refitting.

### RQ1 headline — the E3-vs-E4 gap is source shift, not model size

| arm | setting | macro recall | macro FPR | macro PR-AUC |
|---|---|---:|---:|---:|
| mmBERT-**base** | E3 in-pool test | 0.8890 | 0.0056 | 0.9814 |
| svm | E3 in-pool test | 0.8098 | 0.0097 | 0.9773 |
| mmBERT-**small** | E4 LOSO | 0.6041 | 0.2178 | 0.5701 |
| svm | E4 LOSO | 0.4029 | 0.1678 | 0.5473 |

The in-pool number is high (PR-AUC 0.98 for base) and the held-out-source
number collapses to 0.57 (small) and 0.55 (svm). That is the RQ1 result: the
pool test split overstates generalisation, and the drop is driven by source
shift, not by the transformer. **E3 is base, E4 is small.** There is no
base LOSO arm, so the table cannot separate "base is better than small" from
"in-pool is easier than LOSO"; do not attribute the gap to parameter count.

Frozen embeddings: the input token-embedding matrix (about 197M of the base
model's ~307M parameters) has `requires_grad=False`; the transformer layers and
the classification head train. That is what makes a 307M-parameter model fit in
6 GB, and it also means the SMS/Hinglish token representations are never
adapted, only the layers above them. It is a plausible contributor to the
out-of-source brittleness, not something this phase measured.

### Surprises

1. Gold FPR ~0.49-0.54 for both models, against ~0.006-0.01 on pool test.
   Ranking survives (PR-AUC 0.85 base on v2); the threshold does not.
2. The false positives concentrate in legitimate bank/service messages
   (`bank_alert`, `utility`), not in ordinary personal ham.
3. Conformal coverage is nominal in-pool and collapses under LOSO, including
   ham coverage of 0.17 on `moz`. Abstention rises but does not rescue coverage.
4. ECE is tiny in-pool mainly because `cal_b` is 85% spam and predictions are
   bimodal; this is not evidence of good mid-range calibration.

### What was NOT run

- No mmBERT-base LOSO. E4 covers svm and mmBERT-small only.
- No gold-driven tuning, threshold change or model selection. Gold stayed
  closed for decisions; v2 was scored once.
- No re-scoring of the pool test split for E12-E14. E12 is arithmetic from
  frozen JSON; E13/E14 use the calibration split; E14 LOSO is the disclosed
  deterministic re-score of frozen checkpoints.
- No E14 resplits for LOSO (frozen `q_hat`).
- No bootstrap intervals on gold; n=58 is small and the brief did not ask for
  them.
- No E7-E10 evasion work, no E11 Hinglish slice, no serving, packaging or live
  URL.
- No audit of how realistic the synthetic gold hams are.
- Phase 1 outputs are not committed.

## 2026-11-21 — gold bootstrap CIs, E-recal, E7-E10 evasion

Text-free entry. No message text, gold row, id or note appears here or in any
output file. Gold was not used for tuning or selection.

### Gold bootstrap CIs — `reports/results/e6_gold_v2_bootstrap.json`

Percentile bootstrap, 1,000 resamples, fixed seed 2026. Frozen models were
re-scored with no refit because per-row gold scores were not saved; this is a
deterministic repeat of the gold pass, not a new look. Aggregates only.

| gold | model | n_pos / n_neg | recall [95% CI] | FPR [95% CI] | PR-AUC [95% CI] |
|---|---|---:|---|---|---|
| v1 | svm | 74 / 59 | 0.7162 [0.6103, 0.8195] | 0.4915 [0.3548, 0.6207] | 0.7479 [0.6500, 0.8528] |
| v1 | mmBERT-base | 74 / 59 | 0.8784 [0.7887, 0.9481] | 0.5424 [0.4151, 0.6740] | 0.8767 [0.8112, 0.9282] |
| v1 | dummy | 74 / 59 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.5564 [0.4737, 0.6391] |
| v2 | svm | 23 / 35 | 0.6957 [0.5000, 0.8696] | 0.5429 [0.3684, 0.7059] | 0.6850 [0.5114, 0.8479] |
| v2 | mmBERT-base | 23 / 35 | 1.0000 [1.0000, 1.0000] | 0.4857 [0.3233, 0.6452] | 0.8483 [0.7175, 0.9419] |
| v2 | dummy | 23 / 35 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.3966 [0.2586, 0.5177] |

**v2 recall is degenerate.** mmBERT-base flags 23/23 positives, so every
bootstrap resample gives recall 1.0 and the interval is [1.0, 1.0]. That is a
property of a tiny, easy positive set, not a precise estimate. The v2 FPR CI
is 35 ham rows wide and must be read as such. The dummy PR-AUC is the base-rate
line for each gold set (v1 0.5564, v2 0.3966); every model PR-AUC sits above
it, so ranking survives even where the threshold does not.

### E-recal — `reports/results/e_recal.json`

Recalibrate `tau` on `k` labelled target-domain ham under LOSO, then evaluate
on the remaining held-out rows. This is a descriptive stress test, **not a
zero-shot result**: it uses labelled ham from the held-out source. `k` in
{25, 50, 100}, 20 resplits, seed 42. The svm fold pipeline is refit with the
E4 config and seed because no fold checkpoint was saved; mmBERT-small is
re-scored from the frozen E4 checkpoints with no refit. No held-out row is used
for model fitting.

| model | held out | frozen recall / FPR | k=25 recall / FPR | k=50 recall / FPR | k=100 recall / FPR |
|---|---|---:|---:|---:|---:|
| svm | mishra | 0.4871 / 0.0245 | 0.5968 / 0.0673 | 0.4664 / 0.0322 | 0.4170 / 0.0240 |
| svm | moz | 0.3187 / 0.3111 | 0.0309 / 0.0529 | 0.0083 / 0.0270 | 0.0073 / 0.0240 |
| mmBERT-small | mishra | 0.4957 / 0.0545 | 0.4703 / 0.0630 | 0.4011 / 0.0350 | 0.3558 / 0.0213 |
| mmBERT-small | moz | 0.7125 / 0.3812 | 0.0891 / 0.0560 | 0.0138 / 0.0204 | 0.0053 / 0.0200 |

Reading:

- **Recalibrating on labelled target ham does not fix `moz`.** Recall collapses
  to about 0.01 at about 2% realised FPR (svm 0.0073 at k=100; small 0.0053).
  That is close to base-rate behaviour for `moz` (prevalence 0.234): the score
  ranking is not usable at this operating point.
- On `mishra` the change is roughly neutral: the k=100 FPR is back near the
  frozen FPR (svm 0.0240 vs 0.0245; small 0.0213 vs 0.0545) but recall does not
  improve (svm 0.4170 vs 0.4871; small 0.3558 vs 0.4957).
- The calibration FPR is ~1% by construction (1/k), but the **realised FPR
  stays above 1% even at k=100** on every fold. The quantile does not transfer.
- Never describe this as a fix. It is a diagnostic that shows the threshold,
  not just the ranking, is the failure mode.

### E7-E10 — `reports/results/e7_e10_evasion.json`

One run. Frozen models, thresholds unchanged. E7 = normaliser OFF at inference,
E8 = normaliser ON. Attacks are applied to spam only. E10 masks every URL and
scores the link-less spam slice separately. Aggregates only.

The E8 baseline is asserted against the E3 test macro (unweighted mean over
`mishra` + `moz`) at the frozen E3 threshold. Both models reproduce the E3
recall and FPR to `1e-6` (svm 0.809779 / 0.009650; mmBERT-base 0.889030 /
0.005602). An identity attack (`url_obfuscation` with `p=0`) reproduces clean
recall exactly for both views. `has_url` is a bool column on every pool row.

Pool-wide baseline (all test spam / ham, not the E3 macro):

| model | view | recall | FPR |
|---|---|---:|---:|
| svm | off | 0.9586 | 0.0134 |
| svm | on | 0.9696 | 0.0201 |
| mmBERT-base | off | 0.9807 | 0.0112 |
| mmBERT-base | on | 0.9806 | 0.0112 |

Spam recall under attack, mean ± std over seeds 0/1/2 (n = 5,194 spam):

| model | view | spacing | leet | homoglyph | url obfuscation |
|---|---|---:|---:|---:|---:|
| svm | off | 0.9433 ± 0.0002 | 0.7792 ± 0.0016 | 0.9478 ± 0.0024 | 0.9584 ± 0.0000 |
| svm | on | 0.9560 ± 0.0006 | 0.8392 ± 0.0012 | 0.9804 ± 0.0003 | 0.9698 ± 0.0000 |
| mmBERT-base | off | 0.9671 ± 0.0027 | 0.9480 ± 0.0022 | 0.9740 ± 0.0005 | 0.9807 ± 0.0000 |
| mmBERT-base | on | 0.9670 ± 0.0026 | 0.9490 ± 0.0022 | 0.9802 ± 0.0002 | 0.9806 ± 0.0000 |

E10 (pool test, n = 6,090; link-less spam n = 5,016, all positive):

| model | view | n | recall | FPR |
|---|---|---:|---:|---:|
| svm | normal | 6,090 | 0.9696 | 0.0201 |
| svm | url_masked | 6,090 | 0.9663 | 0.0134 |
| svm | spam_linkless | 5,016 | 0.9705 | — |
| mmBERT-base | normal | 6,090 | 0.9806 | 0.0112 |
| mmBERT-base | url_masked | 6,090 | 0.9790 | 0.0145 |
| mmBERT-base | spam_linkless | 5,016 | 0.9805 | — |

Reading:

- Leetspeak is the strongest single attack for svm: recall falls 0.9586 →
  0.7792 with the normaliser OFF, and the normaliser recovers only to 0.8392.
  mmBERT-base is much less sensitive (0.9807 → 0.9480 / 0.9490).
- For svm, the normaliser ON raises clean recall (0.9586 → 0.9696) and raises
  FPR slightly (0.0134 → 0.0201). Homoglyph recall improves because the
  normaliser folds the look-alikes.
- URL masking does not rescue a link-less slice: link-less spam recall is
  already 0.9705 (svm) and 0.9805 (base), so a URL is not required for the
  model to fire.
- `spam_linkless` has no negatives, so FPR is undefined (`nan`), not zero.

### F1-F5 pre-registration (written before the F run)

Svm only. No gold. Folds: `mishra`, `moz`, `exais`. Macro is the unweighted
mean over `mishra` + `moz`; ExAIS FPR is separate and never enters the macro.
Each arm gets its own train-OOF 1% FPR threshold.

Arms:

- `baseline` — the E4 svm path, guard only.
- `prevalence_50`, `prevalence_20`, `prevalence_05` — class reweighting to an
  effective 50% / 20% / 5% spam training prevalence (`class_weight=None` plus
  explicit sample weights), refit, LOSO.
- `source_class_balanced` — equal total weight per (source, class) cell,
  refit, LOSO.
- `format_masked` — digits→0, currency/phone/URL masks, case-fold; refit on the
  masked view with normalisation off, LOSO; plus a rerun of the E5
  source-vs-rest format AUC on the masked view.
- `F1` — no refit: at the baseline frozen threshold, FP rate by the coarse
  code-based ham category (bank-OTP, transactional, promo, chatty). Counts only,
  no text, no row ids.
- `F5` — diagnostic, no refit: Platt-calibrated scores on train OOF, EM prior
  estimate on each held-out source, prior-adjusted threshold. Assumes `p(x|y)`
  is fixed.

Success rule, fixed before any arm runs: **macro LOSO FPR at the arm's own
frozen 1% threshold must fall by at least 0.05 versus the baseline, while macro
LOSO PR-AUC may fall by at most 0.02.** All arms are reported, including
failures.

Guard, checked before any other arm: the baseline arm through this module's
own code path must reproduce the E4 svm macro (recall 0.402875, FPR 0.167811,
PR-AUC 0.547261) to `1e-6`. If it does not, the run stops and nothing else is
reported.

Time-box: if one arm takes more than 15 minutes of compute, it is rerun on
`moz` + `mishra` only and that is stated. F6 (class-conditional conformal plus
k=100 local calibration) and F7 (unfrozen mmBERT-small on `moz`, stop at 15
minutes) are optional and run only if F1-F5 finish early; if skipped, that is
stated. No arm touches gold.


### F1-F5 results — `reports/results/f_experiments.json`

The guard passed exactly: the baseline arm through this module reproduced the
E4 svm macro recall 0.402875, FPR 0.167811 and PR-AUC 0.547261 to `1e-6`, so
the other arms ran.

| arm | macro recall | macro FPR | macro PR-AUC | Δ FPR | Δ PR-AUC | ExAIS FPR | success |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.4029 | 0.1678 | 0.5473 | +0.0000 | +0.0000 | 0.3811 | — |
| prevalence_50 | 0.3836 | 0.1771 | 0.5416 | +0.0093 | -0.0057 | 0.3738 | no |
| prevalence_20 | 0.3565 | 0.1976 | 0.5302 | +0.0298 | -0.0171 | 0.3146 | no |
| prevalence_05 | 0.3367 | 0.2167 | 0.5062 | +0.0489 | -0.0410 | 0.2700 | no |
| source_class_balanced | 0.0366 | 0.0006 | 0.6840 | -0.1673 | +0.1368 | 0.1442 | yes |
| format_masked | 0.2707 | 0.2007 | 0.5215 | +0.0329 | -0.0258 | 0.3841 | no |

Reading:

- **F2 (prevalence reweighting) fails at every target.** Lowering the training
  prevalence raises macro FPR (0.1678 → 0.1771 / 0.1976 / 0.2167) and lowers
  macro PR-AUC (0.5473 → 0.5416 / 0.5302 / 0.5062). It also lowers ExAIS FPR,
  but not enough to matter, and the both-class operating point gets worse.
- **F3 (source × class balanced) passes the letter of the rule and fails the
  intent.** Macro FPR falls by 0.1673 and PR-AUC rises by 0.1368, but macro
  recall collapses to 0.0366 (mishra 0.0733, moz 0.0000). ExAIS FPR falls to
  0.1442, also at the cost of recall. The pre-registered rule had no recall
  floor, so this arm is reported as a rule-pass with a collapsed operating
  point, **not as a fix**. This is the main surprise of F.
- **F4 (format-masked view) lowers the format shortcut but worsens LOSO.** The
  E5 source-vs-rest format AUC falls (IMC'25 0.9642 → 0.8897; ExAIS 0.9471 →
  0.9089; pooled 0.9779 → 0.9547), so the masks did remove format information.
  LOSO nevertheless gets worse: macro recall 0.4029 → 0.2707, FPR 0.1678 →
  0.2007, PR-AUC 0.5473 → 0.5215. Format cues were not the main cause of the
  source shift; removing them removes signal too.
- **F1: false positives are not only bank messages.** At the baseline frozen
  threshold, `moz` flags 5/5 bank-OTP ham, 9/11 promo ham and 538/1,753 chatty
  ham; ExAIS flags 249/315 transactional ham and 27/39 bank-OTP ham; `mishra`
  flags 8/352 chatty ham. Counts only; no text or ids.
- **F5: EM prevalence is close on `mishra`, badly wrong on `moz` and `exais`.**
  Estimated vs true: mishra 0.3895 vs 0.3873; moz 1.0000 vs 0.2343; exais
  0.8274 vs 0.0000. The prior-adjusted recall/FPR are identical to the frozen
  values on every fold because the adjustment is a monotone transform and the
  threshold is a quantile: it changes the probability cutoff, not the decision.
  It assumes `p(x|y)` is fixed and is a diagnostic, not a fix.

| held out | true prevalence | source prevalence | EM prevalence | frozen recall / FPR | adjusted recall / FPR |
|---|---:|---:|---:|---|---|
| mishra | 0.3873 | 0.8620 | 0.3895 | 0.4871 / 0.0245 | 0.4871 / 0.0245 |
| moz | 0.2343 | 0.9039 | 1.0000 | 0.3187 / 0.3111 | 0.3187 / 0.3111 |
| exais | 0.0000 | 0.9233 | 0.8274 | — / 0.3811 | — / 0.3811 |

Time-box: no arm exceeded 15 minutes. The slowest was `prevalence_05` at
191.3 s, so every arm kept all three folds. F6 and F7 were skipped as optional
because F1-F5 already took about 18 minutes in total; no F6/F7 result exists.
No gold row was read or scored.

## 2026-11-22 — deployment environment check

`make deploy-env` builds a clean venv from `requirements.txt` only, moves the pool
CSVs aside (restored byte-identically, sha256 checked), scores one fixed synthetic
message under both the clean interpreter and the project `.venv`, runs the
Streamlit AppTest, and smokes the Docker image. It writes
`reports/results/deploy_env.json`; the generated section is
"Deployment-environment check" in `docs/DEPLOYMENT.md`. No message text is read,
printed or written.
