# smishlens

**How much of an SMS smishing detector survives when the messages are not
from 2011?** This repository builds a modern, multi-source SMS spam/smishing
pool, measures ranking and threshold portability across sources, and ships a
small Streamlit triage demo on top of a frozen linear model.

The short answer: **the ranking transfers better than the threshold.** A
model that looks excellent in-pool degrades sharply when a whole source is
held out, and the frozen 1% FPR threshold does not port. The demo is
deliberately a triage aid, not a verdict.

---

## 1. Problem

Classical SMS filters report accuracy above 98% on the UCI SMS Spam
Collection, a 2011 UK/Singapore corpus with no parcel scams, no
bank-impersonation lures, no Hinglish, and no modern shortener domains.
Three questions follow:

1. Does a model trained on a modern multi-source pool rank new messages well?
2. Does its operating point (a threshold chosen for 1% FPR) transfer to a
   source it has never seen?
3. Can a simple, frozen display rule turn a score into a usable triage hint
   without pretending to be a verdict?

## 2. Data

The modern pool is frozen by `data/build_modern_pool.py`: exact and
near-duplicate removal, then a stratified 64/16/20 train/calibration/test
split keyed by `(source, label)`. Processed splits are committed; raw
message text is git-ignored.

| Source | Pool rows | ham / positive | Licence status | Role |
|---|---:|---|---|---|
| IMC'25 smishing | 24,462 | 3 / 24,459 | CC BY 4.0 | train / test positives |
| Mishra & Soni | 599 | 367 / 232 | CC BY 4.0 | LOSO fold |
| MOZ-Smishing | 2,330 | 1,784 / 546 | **unverified** (card carries a *model* licence) | LOSO fold |
| ExAIS SMS Spam | 2,330 | 2,330 / 0 | **unverified** (no licence file) | FPR-only LOSO fold |
| Smishtank Dataset I | 726 | 0 / 726 | **unverified** (cite only) | recall-only |
| UCI SMS Spam | — | — | CC BY 4.0 | Tier 1 anchor |

Two properties dominate every aggregate below:

- **Positive-heavy.** Pool test prevalence is 0.8529 on 6,090 rows, so accuracy is not reported.
- **One source dominates.** IMC'25 supplies 94.20% of test spam. Pool-wide evasion numbers are mostly IMC'25 numbers.

Sources with unverified licences are used locally and are never
redistributed; their text does not enter git. See `data/README.md` and
`data/cards/`.

## 3. Method

1. **Normalise** — NFKC, strip format/zero-width characters, fold mixed-script
   confusables, defang URLs and phone numbers.
2. **Features** — word 1–2 gram TF-IDF + character 3–5 gram TF-IDF + a small
   meta block (length, digit ratio, caps ratio, URL/shortcode/currency flags).
3. **Classify** — LinearSVC (`C=0.5`, class-balanced), trained on the pool
   **train** split only.
4. **Threshold** — max recall subject to out-of-fold FPR ≤ 1%, computed on
   train OOF scores, never on test.
5. **Calibrate** — Platt (sigmoid) fitted on `cal_a`.
6. **Conformal** — class-conditional (Mondrian) `q_hat` on the same `cal_a`,
   `alpha = 0.05`.
7. **Display** — three bands derived only from the conformal set:
   `{1}` → "likely scam", `{0}` → "looks ordinary", `{0,1}` or `{}` →
   "needs review". No new thresholds in the app.

No test row and no gold row is used to fit, calibrate, select or tune
anything. The synthetic gold sets are used once, for evaluation.

## 4. Results

All numbers are generated from `reports/results/*.json` by `make summary`
→ `docs/RESULTS_SUMMARY.md`. The headline tables:

### In-pool test (E3, frozen threshold)

| scope | recall | FPR | PR-AUC |
|---|---:|---:|---:|
| `macro` | 0.8098 | 0.0097 | 0.9773 |
| `imc25` | 0.9753 | — | — |
| `smishtank` | 0.9034 | — | — |
| `exais` | — | 0.0322 | — |

`macro` is the unweighted mean over `mishra` + `moz`; `exais` is FPR-only,
`imc25` and `smishtank` are recall-only.

### Leave-one-source-out (E4, frozen threshold)

| held out | recall | FPR | PR-AUC |
|---|---:|---:|---:|
| `mishra` | 0.4871 | 0.0245 | 0.8616 |
| `moz` | 0.3187 | 0.3111 | 0.2329 |
| `exais` | — | 0.3811 | — |
| `imc25` | 0.1550 | — | — |
| `smishtank` | 0.7259 | — | — |
| `macro` | 0.4029 | 0.1678 | 0.5473 |

The gap between the two tables is the whole story: **ranking** mostly holds (`mishra` PR-AUC 0.8616), **threshold** portability does not (`moz` FPR 0.3111).

### Synthetic gold v2 (Claude-authored, 23 spam / 35 ham)

| model | recall | FPR | PR-AUC |
|---|---:|---:|---:|
| svm | 0.6957 | 0.5429 | 0.6850 |
| mmBERT-base | 1.0000 | 0.4857 | 0.8483 |
| dummy (prior) | 0.0000 | 0.0000 | 0.3966 |

The intervals are wide (35 ham rows) and the absolute FPR is not a field estimate. Gold is synthetic; it is never used for selection.

### Evasion (E7/E8, pool test, frozen models)

Normaliser OFF → ON, svm: clean recall 0.9586 → 0.9696, FPR 0.0134 → 0.0201. Leetspeak is the costliest attack (0.7792 → 0.8392).

| model | source | n spam | clean off | clean on | leet off | leet on |
|---|---|---:|---:|---:|---:|---:|
| svm | imc25 | 4,893 | 0.9716 | 0.9753 | 0.8029 | 0.8532 |
| svm | mishra | 47 | 0.5319 | 0.7021 | 0.3901 | 0.6099 |
| svm | moz | 109 | 0.8440 | 0.9174 | 0.1621 | 0.3456 |
| svm | smishtank | 145 | 0.7448 | 0.9034 | 0.5586 | 0.7609 |
| mmBERT-base | imc25 | 4,893 | 0.9826 | 0.9826 | 0.9662 | 0.9671 |
| mmBERT-base | mishra | 47 | 0.8085 | 0.7872 | 0.7234 | 0.7234 |
| mmBERT-base | moz | 109 | 0.9908 | 0.9908 | 0.2049 | 0.2049 |
| mmBERT-base | smishtank | 145 | 0.9655 | 0.9655 | 0.9517 | 0.9517 |

### Fix attempts (F1–F5)

A pre-registered rule required macro LOSO FPR to fall by at least 0.05
while macro PR-AUC fell by at most 0.02. **No arm is a fix.**

- `prevalence_*` makes both FPR and PR-AUC worse.
- `source_class_balanced` (F3) meets the letter of the rule (FPR 0.1678 → 0.0006) only by collapsing macro recall to 0.0366. On `mishra` that is an unweighted-quantile artifact. On `moz` the F3 PR-AUC is 0.4405 against a base rate of 0.2343 — above base rate — but recall is 0.0000 at the 1%-FPR threshold: ranking weak at the top, operating point unusable.
- `format_masked` removes a format shortcut but worsens LOSO (PR-AUC 0.5473 → 0.5215).

## 5. Failure analysis

- **`moz` shows a language-shaped FPR gap, but the check is partial.** Source metadata tags 100.00% of `moz` `pt`. The stopword heuristic (not a language-ID model) puts 18.54% in `en` and 81.33% in `other`; the `en` bucket in a 100% `pt` source is short or ambiguous text, not real English. Ham FPR is 0.3393 for `other`-looking versus 0.1445 for `en`-looking rows, while `exais` shows no gap (0.3830 `en` versus 0.3744 `other`). This is consistent with language shift on `moz`, not proof of it. At the frozen threshold `moz` recall is 0.3187 and FPR 0.3111; PR-AUC is 0.2329 against a base rate of 0.2343.
- **ExAIS is transactional ham.** Under LOSO, transactional ham FPR is 0.7905 and bank-OTP ham FPR is 0.6923. The model reads invoice and alert language as spammy.
- **Synthetic gold false positives are legitimate bank/service messages.** svm FPR is 0.5429 on 35 ham rows. The pool contains almost no real bank-alert ham, so this is a data gap as much as a model gap.
- **Recalibration on labelled target ham does not fix `moz`.** At k=100, mean recall is 0.0073 at a realised FPR of 0.0240.
- **F3 passes the rule only through collapsed recall.** On `mishra` the collapse is the unweighted OOF quantile (a weighted quantile recovers 0.2328 recall). On `moz` the F3 PR-AUC is 0.4405 against a base rate of 0.2343 — above base rate — but recall is 0.0000 at the 1%-FPR threshold: ranking weak at the top, operating point unusable. No arm is a fix.
- **Pool-wide aggregates flatter the model.** One source dominates the positives.

## 6. Limitations

- Positive-heavy synthetic pool; not a field sample.
- No real bank-alert ham, and almost no multilingual coverage.
- One message at a time; no conversation context.
- Threshold does not port across sources; ranking does not always either.
- The normaliser's benefit is bought with a higher FPR (0.0134 → 0.0201 on svm).
- mmBERT-base is measured but not hosted; the hosted path is svm only.
- Accuracy is never reported; the base rate would dominate it.

## 7. The demo app

`streamlit run app/app.py` (see `requirements-app.txt` for the minimal
install). The app shows:

- a calibrated P(spam), the conformal set, and one band;
- extracted URL(s) and basic message stats;
- an optional sender-ID field that is displayed only and **does not change
  the score** (experimental);
- an always-visible notice: triage aid, not a verdict; about 54% FPR on synthetic bank/service ham; almost no real bank-alert ham in the pool; weak on non-English held-out sources; nothing stored or logged;
- 10 invented synthetic demo messages, verified to have no near-duplicate in gold or the pool: 0 pairs at the 0.90 char-TF-IDF rule, maximum similarity 0.47 against the pool and 0.46 against gold.

<!-- Screenshot placeholders: replace with real captures before publishing. -->

| Screenshot | File |
|---|---|
| Empty state with the notice | `docs/img/app-empty.png` *(placeholder)* |
| Scored smishing example | `docs/img/app-scam.png` *(placeholder)* |
| "Needs review" example | `docs/img/app-review.png` *(placeholder)* |

mmBERT-base is available only behind `SMISHLENS_ENABLE_MMBERT=1` and is
never required by the hosted app.

## 8. How to run

```bash
git clone <repo> smishlens && cd smishlens
make setup          # venv + deps + UCI anchor + frozen split
make test           # pytest
make app            # Streamlit on http://localhost:8501
make summary        # regenerate all docs from the JSONs
```

Capstone app only (no dev tooling):

```bash
pip install -r requirements-app.txt
python -m smishlens.serve --build    # writes artifacts/serve/ (git-ignored)
streamlit run app/app.py
```

The FastAPI wrapper (`make serve`) and the original UCI model remain for
the Tier 1 pipeline; the capstone app uses `smishlens.serve` and the pool
artifact.

## 9. Repo layout

```
src/smishlens/     library: data, normalise, features, calibrate, conformal,
                   pool_experiments, loso, f_experiments, e7_e10, serve, ...
app/app.py         Streamlit capstone app (svm, frozen bands)
serve/api.py       FastAPI wrapper for the Tier 1 UCI artifact
data/              download/build scripts, cards/, processed splits (raw ignored)
reports/           verification_log.md, results/*.json, figures/
docs/              RESULTS_SUMMARY.md, SLIDES_OUTLINE.md, DEPLOYMENT.md
tests/             pytest suite, including serve and Streamlit AppTest
artifacts/         trained models + manifests (git-ignored; rebuilt locally)
```

Deployment plan: `docs/DEPLOYMENT.md` (public/private split; the public
repository carries no model artifact).

## 10. Licence

Code is MIT (see `LICENSE`). Datasets keep their own licences. Nothing here
re-licenses third-party data, and **no model trained on the unverified-licence
sources is cleared for redistribution**; see `MODEL_CARD.md` and
`docs/DEPLOYMENT.md`. Keeping the artifact in a private repository avoids
redistributing it, but it does not clear the dataset licences for serving
predictions.

_Latency (E17, CPU): svm 2.84 ms p50 per message, 3.8 MB on disk._
