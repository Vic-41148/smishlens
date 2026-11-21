# Slides outline

Ten slides. Every number comes from `reports/results/*.json` and matches
`docs/RESULTS_SUMMARY.md`. No message text appears on any slide.

---

## 1. The problem

- "98% accurate" SMS filters are usually measured on a 2011 corpus.
- How much survives when the messages are modern and from several sources?
- Two claims to test separately: **ranking** quality and **threshold**
  portability.
- Product shape: a triage aid with three display bands, not an auto-blocker.

Speaker note: state up front that the deliverable is a demo, not a verdict.

---

## 2. Data and licences

- Pool test: 6,090 rows, prevalence 0.8529 (positive-heavy by design).
- Sources: IMC'25, Mishra & Soni, MOZ-Smishing, ExAIS, Smishtank, UCI.
- IMC'25 supplies 94.20% of test spam, so pool-wide numbers are dominated by one source.
- Licences: IMC'25 / Mishra / UCI CC BY 4.0; **MOZ, ExAIS and Smishtank
  unverified or missing**. Raw text is never redistributed.
- The pool has almost no real bank-alert ham; that gap drives the failure
  mode on slide 7.

Speaker note: the licence row matters for any public deployment.

---

## 3. Method

- Normalise (NFKC, confusables, defang URLs), then word 1–2 gram + char 3–5
  gram TF-IDF + meta features.
- LinearSVC, class-balanced, trained on the pool **train** split.
- Frozen threshold: max recall s.t. out-of-fold FPR ≤ 1%, from train OOF
  only.
- Platt calibration on `cal_a`; class-conditional conformal `q_hat` on the
  same `cal_a`, `alpha = 0.05`.
- No test or gold row is used to fit, calibrate, select or tune.

---

## 4. In-pool results (E3)

| scope | recall | FPR | PR-AUC |
|---|---:|---:|---:|
| `macro` | 0.8098 | 0.0097 | 0.9773 |
| `imc25` | 0.9753 | — | — |
| `smishtank` | 0.9034 | — | — |
| `exais` | — | 0.0322 | — |

- These are the flattering numbers. The next slide is the honest one.

---

## 5. LOSO and source shift (E4)

| held out | recall | FPR | PR-AUC |
|---|---:|---:|---:|
| `mishra` | 0.4871 | 0.0245 | 0.8616 |
| `moz` | 0.3187 | 0.3111 | 0.2329 |
| `exais` | — | 0.3811 | — |
| `macro` | 0.4029 | 0.1678 | 0.5473 |

- Ranking transfers better than the threshold.
- `moz` PR-AUC 0.2329 against a base rate of 0.2343: the ranking itself is near
chance there. Recall 0.3187, FPR 0.3111.
- This is a whole-source holdout, not a random split.

---

## 6. Synthetic gold (Claude-authored, with bootstrap CIs)

| model | recall [95% CI] | FPR [95% CI] | PR-AUC [95% CI] |
|---|---|---|---|
| svm | 0.6957 [0.5000, 0.8696] | 0.5429 [0.3684, 0.7059] | 0.6850 [0.5114, 0.8479] |
| mmBERT-base | 1.0000 [1.0000, 1.0000] | 0.4857 [0.3233, 0.6452] | 0.8483 [0.7175, 0.9419] |
| dummy (prior) | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.3966 [0.2586, 0.5177] |

- 23 spam / 35 ham. The intervals are wide.
- Gold is synthetic, never used for selection, and its absolute FPR is not
  a field estimate.

---

## 7. Failure analysis

- **`moz` shows a language-shaped FPR gap, but the check is partial.** Source metadata tags 100.00% of `moz` `pt`. The stopword heuristic (not a language-ID model) puts 18.54% in `en` and 81.33% in `other`; the `en` bucket in a 100% `pt` source is short or ambiguous text, not real English. Ham FPR is 0.3393 for `other`-looking versus 0.1445 for `en`-looking rows, while `exais` shows no gap (0.3830 `en` versus 0.3744 `other`). This is consistent with language shift on `moz`, not proof of it. At the frozen threshold `moz` recall is 0.3187 and FPR 0.3111; PR-AUC is 0.2329 against a base rate of 0.2343.
- **ExAIS is transactional ham.** Under LOSO, transactional ham FPR is 0.7905 and bank-OTP ham FPR is 0.6923. The model reads invoice and alert language as spammy.
- **Synthetic gold false positives are legitimate bank/service messages.** svm FPR is 0.5429 on 35 ham rows. The pool contains almost no real bank-alert ham, so this is a data gap as much as a model gap.
- **Recalibration on labelled target ham does not fix `moz`.** At k=100, mean recall is 0.0073 at a realised FPR of 0.0240.
- **F3 passes the rule only through collapsed recall.** On `mishra` the collapse is the unweighted OOF quantile (a weighted quantile recovers 0.2328 recall). On `moz` the F3 PR-AUC is 0.4405 against a base rate of 0.2343 — above base rate — but recall is 0.0000 at the 1%-FPR threshold: ranking weak at the top, operating point unusable. No arm is a fix.
- **Pool-wide aggregates flatter the model.** One source dominates the positives.

---

## 8. Evasion per source (E7/E8/E10)

- Normaliser OFF → ON, svm: clean recall 0.9586 → 0.9696, FPR 0.0134 → 0.0201.
- Pool-wide evasion is dominated by `imc25`; read the per-source table.

Per-source E7/E8/E10 spam recall (imc25 is about 94% of test spam):

| model | source | n spam | E7 off | E8 on | leet off | leet on | E10 linkless |
|---|---|---:|---:|---:|---:|---:|---:|
| svm | imc25 | 4,893 | 0.9716 | 0.9753 | 0.8029 | 0.8532 | 0.9750 |
| svm | mishra | 47 | 0.5319 | 0.7021 | 0.3901 | 0.6099 | 0.6667 |
| svm | moz | 109 | 0.8440 | 0.9174 | 0.1621 | 0.3456 | 0.9174 |
| svm | smishtank | 145 | 0.7448 | 0.9034 | 0.5586 | 0.7609 | 0.7308 |
| mmBERT-base | imc25 | 4,893 | 0.9826 | 0.9826 | 0.9662 | 0.9671 | 0.9825 |
| mmBERT-base | mishra | 47 | 0.8085 | 0.7872 | 0.7234 | 0.7234 | 0.7576 |
| mmBERT-base | moz | 109 | 0.9908 | 0.9908 | 0.2049 | 0.2049 | 0.9908 |
| mmBERT-base | smishtank | 145 | 0.9655 | 0.9655 | 0.9517 | 0.9517 | 0.8462 |

Speaker note: never quote one evasion percentage without naming the source
mix.

---

## 9. Latency, footprint, and the app

Synthetic probe strings, CPU only; each arm runs in its own subprocess:

| arm | batch 1 p50 | batch 1 p95 | batch 32 p50 / msg | batch 32 p95 / msg | disk MB | peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| svm | 2.84 ms | 3.01 ms | 0.44 ms | 0.45 ms | 3.8 | 227 |
| mmBERT-base | 54.29 ms | 59.10 ms | 32.02 ms | 32.04 ms | 1205.9 | 1610 |

- Hosted path is svm only; mmBERT-base is optional and local-only.
- Output: calibrated score, conformal set, and one band: "likely scam" /
  "looks ordinary" / "needs review". Bands come only from the frozen
  conformal rule; no new thresholds.
- Artifact on disk: 3.8 MB; peak RSS
  227 MB.
- Always-visible notice: triage aid, not a verdict; nothing stored or logged.

---

## 10. Limitations and deployment

- **Limits:** positive-heavy synthetic pool; one source dominates; threshold
  does not port; no real bank-alert ham; no multilingual coverage; single
  message, no context; accuracy is never reported.
- **Two repositories, one clone.** `origin` is public (code, docs, text-free
  data). The serving bundle lives only on a local `private-deploy` branch
  pushed to a `private` remote you add.
- The bundle is a TF-IDF vocabulary plus SVC weights derived from pool text,
  so keeping it out of the public repository avoids redistribution.
- Hosted on Streamlit Community Cloud, private repository, entrypoint
  `app/app.py`, Python 3.12.14 from `deploy_env.json`.
- **Reproduce:** `make summary`, `make build-serve`,
  `streamlit run app/app.py`. Full plan: `docs/DEPLOYMENT.md`.
