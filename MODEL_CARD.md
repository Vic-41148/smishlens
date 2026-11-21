# Model card — smishlens pool svm triage

## What this is

A binary SMS spam/smishing **ranking model** with a frozen operating point
and a class-conditional conformal display rule. It is packaged for a
Streamlit triage demo, not for automated blocking.

- **Model:** LinearSVC (`C=0.5`, `class_weight="balanced"`, `dual="auto"`, seed 42)
  over word 1–2 gram + character 3–5 gram TF-IDF and a small meta block.
- **Trained on:** the modern pool **train** split only.
- **Calibration:** Platt (sigmoid) fitted on `cal_a`.
- **Conformal:** class-conditional (Mondrian) `q_hat` fitted on the same `cal_a`,
  `alpha = 0.05`.
- **Frozen threshold:** `tau` from the E3 recipe (max recall s.t. out-of-fold
  FPR ≤ 1%), unchanged at serving time.
- **No test row and no gold row is used to fit, calibrate or select anything.**

## Intended use

- Research and demonstration of a triage-aid UX: score, conformal set, and one
  of three display bands.
- Comparing ranking quality and threshold portability across SMS sources.
- Teaching material about why a high in-pool score does not transfer.

## Out of scope

- **Not a verdict.** The band is a triage hint. Do not auto-block, auto-delete
  or auto-report a message from it.
- **Not a bank-alert filter.** The pool contains almost no real bank-alert ham, and the model flags 54% of legitimate bank/service messages in synthetic gold.
- **Not multilingual.** The pool is overwhelmingly English; held-out
  non-English sources degrade badly.
- **Not a sender-reputation system.** The app shows a sender-ID field for
  context only. It does not read it, and it does not change the score.
- **Not a conversation model.** One message at a time, no thread context.

## Data and licences

Only scripts and processed splits are committed. Raw message text is
git-ignored.

| Source | Pool rows | ham / positive | Licence status | Role |
|---|---:|---|---|---|
| IMC'25 smishing | 24,462 | 3 / 24,459 | CC BY 4.0 | train / test positives |
| Mishra & Soni | 599 | 367 / 232 | CC BY 4.0 | LOSO fold |
| MOZ-Smishing | 2,330 | 1,784 / 546 | **unverified** (card carries a *model* licence) | LOSO fold |
| ExAIS SMS Spam | 2,330 | 2,330 / 0 | **unverified** (no licence file) | FPR-only LOSO fold |
| Smishtank Dataset I | 726 | 0 / 726 | **unverified** (cite only) | recall-only |
| UCI SMS Spam | — | — | CC BY 4.0 | Tier 1 anchor |

MOZ, ExAIS and Smishtank have **unverified or missing data licences**.
Their text is never redistributed here. A model trained on them is a derived
artefact; redistributing the weights is a separate legal question and is
**not cleared** by this card. The deployment split keeps the artifact in a
private repository; see `docs/DEPLOYMENT.md`.

## Evaluation

Numbers are read from `reports/results/*.json`; see
`docs/RESULTS_SUMMARY.md` for the full tables.

### In-pool test (E3, frozen threshold)

| scope | recall | FPR | PR-AUC |
|---|---:|---:|---:|
| `macro` | 0.8098 | 0.0097 | 0.9773 |
| `imc25` | 0.9753 | — | — |
| `smishtank` | 0.9034 | — | — |
| `exais` | — | 0.0322 | — |

The pool test prevalence is 0.8529, so accuracy is not reported.

### Leave-one-source-out (E4, frozen threshold)

| held out | recall | FPR | PR-AUC |
|---|---:|---:|---:|
| `mishra` | 0.4871 | 0.0245 | 0.8616 |
| `moz` | 0.3187 | 0.3111 | 0.2329 |
| `exais` | — | 0.3811 | — |
| `imc25` | 0.1550 | — | — |
| `smishtank` | 0.7259 | — | — |
| `macro` | 0.4029 | 0.1678 | 0.5473 |

Ranking transfers better than the threshold. `moz` is the failure case: PR-AUC 0.2329 against a base rate of 0.2343, i.e. near chance.

### Synthetic gold v2 (Claude-authored, not field data)

| model | recall | FPR | PR-AUC |
|---|---:|---:|---:|
| svm | 0.6957 | 0.5429 | 0.6850 |
| mmBERT-base | 1.0000 | 0.4857 | 0.8483 |
| dummy (prior) | 0.0000 | 0.0000 | 0.3966 |

Only 35 ham rows. The FPR interval is wide, and the absolute value is not a field estimate.

### Evasion (E7/E8, pool test, frozen models)

Normaliser OFF vs ON, svm: clean recall 0.9586 → 0.9696, FPR 0.0134 → 0.0201. Leetspeak is the costliest attack (0.7792 → 0.8392). Pool-wide numbers are dominated by `imc25`; the per-source breakdown is in `docs/RESULTS_SUMMARY.md`.

### Latency and footprint (E17, CPU, synthetic probes)

svm: 2.84 ms p50 / 3.01 ms p95 per message, 3.8 MB on disk, 227 MB peak RSS. mmBERT-base: 54.29 ms p50 per message, 1205.9 MB on disk, 1610 MB peak RSS. That is why mmBERT-base is optional and local-only.

## Known failures

- **`moz` shows a language-shaped FPR gap, but the check is partial.** Source metadata tags 100.00% of `moz` `pt`. The stopword heuristic (not a language-ID model) puts 18.54% in `en` and 81.33% in `other`; the `en` bucket in a 100% `pt` source is short or ambiguous text, not real English. Ham FPR is 0.3393 for `other`-looking versus 0.1445 for `en`-looking rows, while `exais` shows no gap (0.3830 `en` versus 0.3744 `other`). This is consistent with language shift on `moz`, not proof of it. At the frozen threshold `moz` recall is 0.3187 and FPR 0.3111; PR-AUC is 0.2329 against a base rate of 0.2343.
- **ExAIS is transactional ham.** Under LOSO, transactional ham FPR is 0.7905 and bank-OTP ham FPR is 0.6923. The model reads invoice and alert language as spammy.
- **Synthetic gold false positives are legitimate bank/service messages.** svm FPR is 0.5429 on 35 ham rows. The pool contains almost no real bank-alert ham, so this is a data gap as much as a model gap.
- **Recalibration on labelled target ham does not fix `moz`.** At k=100, mean recall is 0.0073 at a realised FPR of 0.0240.
- **F3 passes the rule only through collapsed recall.** On `mishra` the collapse is the unweighted OOF quantile (a weighted quantile recovers 0.2328 recall). On `moz` the F3 PR-AUC is 0.4405 against a base rate of 0.2343 — above base rate — but recall is 0.0000 at the 1%-FPR threshold: ranking weak at the top, operating point unusable. No arm is a fix.
- **Pool-wide aggregates flatter the model.** One source dominates the positives.

## Operational notes for the demo

- The app shows an always-visible notice: triage aid, not a verdict; about 54% FPR on synthetic bank/service ham; almost no real bank-alert ham in the pool; weak on non-English held-out sources; nothing stored or logged.
- The artifact is hash-manifested and refused at load time if it changed.
- Bands are derived only from the frozen conformal set:
  `{1}` → "likely scam", `{0}` → "looks ordinary", `{0,1}` or `{}` →
  "needs review". No new thresholds are introduced in the app.

## Maintenance

- Rebuild the artifact with `python -m smishlens.serve --build` (pool train +
  `cal_a` only).
- Regenerate all reported numbers with `make summary`.
- Any change to the recipe requires a new model card and new evaluation
  numbers.
