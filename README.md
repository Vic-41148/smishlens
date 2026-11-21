# smishlens

An SMS spam/smishing detector built to answer one question honestly: **how much of a
"98% accurate" SMS filter survives when the messages are not from 2011?** Tier 1 is the
classical pipeline on the UCI anchor; Tier 2 moves to a modern, multi-source pool.

## Three actions, not two

A binary "spam / not spam" gate is the wrong product. The policy layer returns one of three
actions, and it is deliberately FPR-first: a filtered security advisory is worse than a
delivered advertisement.

| Action | Meaning | When |
|---|---|---|
| `deliver` | show it | the conformal set includes 0, or the signals disagree |
| `quarantine` | hold for review, do not notify | the conformal set is `{0,1}` (ambiguous) or `{}` (unlike either class) |
| `junk` | block / hide | the conformal set is `{1}` **and** `p_spam >= tau` |

On the UCI test split this gives **874 deliver / 129 junk / 30 quarantine** (abstain rate
0.0290) at `tau = 0.1514`.

## Results — UCI test split, touched once

From `reports/results/test_metrics.json`. 1,033 rows (128 spam / 905 ham, prevalence 0.1239).
Bootstrap 95% intervals, 3 dp:

| Metric | Value | 95% CI |
|---|---|---|
| Recall (TPR) | 0.977 | 0.946 – 1.000 |
| Precision | 0.969 | 0.935 – 0.992 |
| F1 | 0.973 | 0.951 – 0.989 |
| PR-AUC | 0.994 | 0.986 – 0.999 |
| ROC-AUC | 0.999 | — |
| FPR at `tau` | 0.0044 | — |

**Read the intervals, not the point estimates.** 128 spam rows is a small sample; the recall
interval reaches 1.000 and the lower bounds are what a reviewer should plan against.

Champion model: `word_char_meta_linearsvc_sigmoid` (v1.0.0), dataset hash `f97f67a049e40da3`.

## How it works

1. **Normalise** — NFKC, strip format/zero-width characters, fold confusables, defang URLs and
   phone numbers so a link cannot be memorised as a token.
2. **Features** — word 1–2 gram TF-IDF + character 3–5 gram TF-IDF + a small meta block (length,
   digit ratio, caps ratio, URL/shortcode/currency flags). The train-split ablation
   (`reports/results/tier1_e2.csv`) puts word+char at PR-AUC 0.9848 ± 0.0044, ahead of either
   alone.
3. **Classify** — LinearSVC, sigmoid-calibrated (temperature fitted on train), so the score is a
   usable probability. LinearSVC was best on the train-split comparison at PR-AUC 0.9845 ± 0.0064
   (`reports/results/tier1_e3.csv`), ahead of LR, CNB, MNB and RF.
4. **Threshold** — `tau` is the max-recall point subject to FPR ≤ 0.01, fitted on **out-of-fold
   train scores**, never on test.
5. **Conformal** — Mondrian conformal per class (`alpha = 0.05`) fitted on the **calibration**
   split. This produces the `{0}`, `{1}`, `{0,1}`, `{}` sets that drive the three actions.
6. **Policy** — `src/smishlens/policy.py` turns (probability, threshold, set) into an action.

## Quickstart

```bash
git clone <repo> smishlens && cd smishlens
make setup      # venv, deps, download UCI, build the frozen split, train if needed
make test       # pytest
make app        # Streamlit demo on http://localhost:8501
make serve      # FastAPI on http://127.0.0.1:8000 (POST /classify, GET /health)
make eda        # regenerate reports/figures/ and reports/eda.md
make data       # fetch every Tier 2 source and rebuild the modern pool locally
```

`make setup` and `make train` **never** touch the test split. `smishlens.final_eval` is a
one-shot script that refuses to overwrite `reports/results/test_metrics.json`; it is not wired
into any target on purpose.

## Repo layout

```
src/smishlens/    library: data, normalise, features, models, calibrate, conformal,
                  policy, predict, train, final_eval, experiments, eda, loso, probes
serve/api.py      FastAPI wrapper (no logic; reads artifacts/ via ARTIFACT_DIR)
app/app.py        Streamlit demo, same code path as the API
data/             download_data.py, audit_sources.py, build_uci_anchor.py,
                  build_modern_pool.py, cards/ (one DATA_CARD per source), raw/ (ignored)
reports/          eda.md, error_analysis.md, verification_log.md, results/, figures/
tests/            pytest suite (normaliser, data, pipeline, serving, conformal/policy)
artifacts/        trained model + manifest (git-ignored; rebuilt by `make train`)
models/           third-party encoder weights (git-ignored; data/download_models.py)
```

## Data and licences

Only **scripts** and **processed splits** are committed. Raw message text is git-ignored: it
contains PII, and several licences do not permit redistribution. See `data/README.md` for the
full acquisition manifest and `data/processed/uci/DATA_CARD.md` for the anchor card.

| Source | Rows | Labels | Licence | Use |
|---|---|---|---|---|
| UCI SMS Spam Collection v.1 | 5,574 (5,171 unique) | 4,827 ham / 747 spam | CC BY 4.0 | Tier 1 anchor |
| IMC'25 smishing | 33,869 | smishing only | CC BY 4.0 | Tier 2, recall-only |
| Mishra & Soni | 5,971 | ham / spam / smishing | CC BY 4.0 | Tier 2, LOSO |
| MOZ-Smishing | 2,561 | 2,009 / 552 | unclear (card carries a *model* licence) | Tier 2, LOSO |
| ExAIS SMS Spam | 5,240 | 2,890 ham / 2,350 spam | unclear (no licence file) | **ham-only 2026-11-21**: 2,330 ham kept, every spam row dropped ("spam" is bulk/unsolicited, not smishing); FPR-only LOSO fold |
| Smishtank Dataset I | 1,062 | smishing only | unclear (cite only) | Tier 2, recall-only |
| WSPR Lab | 68,029 msgs | phishing only | MIT (repo) | Tier 2, recall-only |
| Africa HF set | 10,000 | 5,000 / 5,000 | unclear | **synthetic — never in a test set** |

Sources with unclear licences are used locally and are never redistributed; their text does not
enter git.

## Evaluation protocol

- **Splits.** 3,301 train / 825 calibration / 1,033 test, stratified, seed 42, materialised once by
  `data/build_uci_anchor.py` and committed. The split is reproducible: rebuilding it from the raw
  corpus yields a byte-identical tree.
- **No test tuning.** Model choice (E1–E3) and threshold selection happen on train with 5-fold CV
  and out-of-fold scores. Conformal quantiles come from calibration. Test is opened once.
- **Metrics.** PR-AUC is primary; F1, MCC, ROC-AUC, recall at FPR ≤ 1%. Accuracy is not reported
  because prevalence is 0.125.
- **Disclosure — the test split was printed once before the final retrain.** An earlier run,
  before defanged-URL masking was added to `normalise.py`, printed test metrics to the console.
  Those numbers were never saved to `reports/results/` and were superseded; no decision was taken
  from them. The single authoritative `test_metrics.json` was written afterwards, once. Full text
  in `reports/verification_log.md`.
- **Errors.** All 7 test errors and 13 conformal abstentions are hand-tagged in
  `reports/error_analysis.md`. The costliest false-positive class is a genuine bank security
  advisory, which is exactly why the policy is FPR-first.

## Limitations

- **UCI is a 2011 UK/Singapore corpus.** It is saturated (PR-AUC ≈ 0.99 for several models) and
  says nothing about modern smishing: no parcel-delivery scams, no bank-impersonation lures, no
  Hinglish, no current shortener domains. Treat every number above as a pipeline check, not a
  field-performance claim.
- **The normaliser is untested here.** On UCI the E1 arms tie within fold noise — raw is nominally
  best by about 0.0003 AP, and the differences are smaller than one fold std. That is expected:
  the anchor has almost no Unicode or defanging obfuscation for the normaliser to repair. Its
  value must be shown on the modern pool and the evasion suite (E7/E8) in Tier 2.
- **Small test sample.** 128 positives; see the intervals.
- **In-pool conformal coverage only.** Test coverage is 0.948 (ham) / 0.977 (spam) at alpha 0.05.
  Coverage under a held-out *source* is the separate RQ5 question and is not established here.
- **No conversation context.** "what is your account number?" is phishing from a stranger and ham
  from a friend; the model sees one message.

## Licence

Code in this repository is MIT (see `LICENSE`). Datasets keep their own licences, listed above
and in `data/README.md`; nothing here re-licenses third-party data. Encoder weights
(`models/`, git-ignored) are third-party checkpoints under their own licences (mmBERT: MIT).
