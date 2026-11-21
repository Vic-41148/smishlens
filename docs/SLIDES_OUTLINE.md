# Slides outline

Ten slides. Every number comes from `reports/results/*.json` and matches
`docs/RESULTS_SUMMARY.md`. No message text appears on any slide.

---

## 1. The question

- "98% accurate" SMS filters are usually measured on a 2011 corpus.
- How much survives when the messages are modern and from several sources?
- Two claims to test separately: **ranking** quality and **threshold**
  portability.
- Product shape: a triage aid with three display bands, not an auto-blocker.

Speaker note: state up front that the deliverable is a demo, not a verdict.

---

## 2. Data

- Pool test: 6,090 rows, prevalence 0.8529 (positive-heavy by design).
- Sources: IMC'25, Mishra & Soni, MOZ-Smishing, ExAIS, Smishtank, UCI.
- IMC'25 supplies 94.20% of test spam, so pool-wide numbers are dominated by one source.
- Licences: IMC'25 / Mishra / UCI CC BY 4.0; **MOZ, ExAIS and Smishtank
  unverified or missing**. Raw text never redistributed.

Speaker note: the licence row matters for any public deployment.

---

## 3. Method

- Normalise (NFKC, confusables, defang URLs), then word 1–2 gram + char 3–5
  gram TF-IDF + meta features.
- LinearSVC, class-balanced, trained on the pool **train** split.
- Frozen threshold: max recall s.t. out-of-fold FPR ≤ 1%.
- Platt calibration on `cal_a`; class-conditional conformal `q_hat` on the
  same `cal_a`, `alpha = 0.05`.
- No test or gold row is used to fit, calibrate or select.

---

## 4. In-pool test (E3)

| scope | recall | FPR | PR-AUC |
|---|---:|---:|---:|
| `macro` | 0.8098 | 0.0097 | 0.9773 |
| `imc25` | 0.9753 | — | — |
| `smishtank` | 0.9034 | — | — |
| `exais` | — | 0.0322 | — |

- These are the flattering numbers. The next slide is the honest one.

---

## 5. Leave-one-source-out (E4)

| held out | recall | FPR | PR-AUC |
|---|---:|---:|---:|
| `mishra` | 0.4871 | 0.0245 | 0.8616 |
| `moz` | 0.3187 | 0.3111 | 0.2329 |
| `exais` | — | 0.3811 | — |
| `macro` | 0.4029 | 0.1678 | 0.5473 |

- Ranking transfers better than the threshold.
- `moz` PR-AUC 0.2329 vs a base rate near 0.2343: the ranking itself is near chance there.

---

## 6. Failure analysis

- **`moz` is a language-shift fold.** The card lists MOZ as Portuguese (Mozambique), and the manifest tags 100.00% of it `pt`. A stopword heuristic (not a language-ID model) puts 81.33% of `moz` rows in "other" versus 8.51% in `mishra`. The highest `moz` ham FPR sits in the non-English bucket (0.3393). At the frozen threshold `moz` recall is 0.3187 and FPR 0.3111; PR-AUC is 0.2329 against a base rate of 0.2343.
- **ExAIS is transactional ham.** Under LOSO, transactional ham FPR is 0.7905 and bank-OTP ham FPR is 0.6923. The model reads invoice and alert language as spammy.
- **Synthetic gold false positives are legitimate bank/service messages.** svm FPR is 0.5429 on 35 ham rows. The pool contains almost no real bank-alert ham, so this is a data gap as much as a model gap.
- **Recalibration on labelled target ham does not fix `moz`.** At k=100, mean recall is 0.0073 at a realised FPR of 0.0240.
- **F3 passes the rule only through collapsed recall.** On `mishra` the collapse is the unweighted OOF quantile (a weighted quantile recovers 0.2328 recall); on `moz` recall stays 0.0000 even with a weighted quantile, so that fold is a ranking failure. No arm is a fix.
- **Pool-wide aggregates flatter the model.** One source dominates the positives.

---

## 7. Evasion (E7/E8/E10)

- Normaliser OFF → ON, svm: clean recall 0.9586 → 0.9696, FPR 0.0134 → 0.0201.
- Leetspeak is the costliest attack: 0.7792 → 0.8392.
- Pool-wide evasion is dominated by `imc25`; the per-source table is in the
  summary. `moz` and `smishtank` behave differently from `imc25`.
- mmBERT-base is less sensitive to homoglyphs and leetspeak than svm.

Speaker note: do not quote a single evasion percentage without naming the
source mix.

---

## 8. Fix attempts (F1–F5): none is a fix

- Pre-registered rule: macro FPR must drop ≥ 0.05 while macro PR-AUC falls
  ≤ 0.02.
- `prevalence_*`: worse FPR and worse PR-AUC.
- `source_class_balanced` (F3): passes the letter of the rule (FPR 0.1678 → 0.0006) **only** by collapsing recall to 0.0366. Post-hoc: unweighted-quantile artifact on `mishra`, ranking failure on `moz`.
- `format_masked`: removes the format shortcut but worsens LOSO (PR-AUC 0.5473 → 0.5215).
- Every arm is reported, including failures.

---

## 9. The app

- Streamlit, svm only on the hosted path; mmBERT-base is optional and
  local-only.
- Output: calibrated score, conformal set, and one band: "likely scam" /
  "looks ordinary" / "needs review".
- Bands come only from the frozen conformal rule; no new tuning.
- Always-visible notice: triage aid, not a verdict; 54% FPR on synthetic bank/service ham; almost no real bank-alert ham; weak on non-English sources; nothing stored or logged.
- Sender-ID field is displayed only and does not change the score
  (experimental).
- Latency: svm 2.84 ms p50 per message.

---

## 10. Limits and next steps

- **Limits:** positive-heavy synthetic pool; one source dominates; threshold
  does not port; no real bank-alert ham; no multilingual coverage; single
  message, no context.
- **What would help:** real bank/service ham, non-English held-out sources,
  and a target-specific calibration set.
- **What not to do:** ship the frozen threshold as an auto-blocker, or quote
  in-pool numbers as field performance.
- **Reproduce:** `make summary`, `python -m smishlens.serve --build`,
  `streamlit run app/app.py`.
