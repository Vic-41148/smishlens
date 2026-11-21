# Demo script (3 minutes)

Generated from `reports/results/*.json` by `make summary`. Use only the
synthetic demo messages built into the app: they are invented for this
demo, are not gold rows, and are not pool rows. Never paste a real
message, and never show a real-alert evaluation row.

Pre-flight: `streamlit run app/app.py`, confirm the always-visible notice
is present, and keep the browser at a readable zoom.

## Run order

| time | action | say |
|---|---|---|
| 0:00–0:20 | Open the app. Point at the notice. | "This is a triage aid, not a verdict. The pool is 85.29% positive, so we never quote accuracy." |
| 0:20–0:50 | Click the demo button **Parcel fee**, then **Check message**. | "A delivery-fee lure with a defanged URL. Band: likely scam; calibrated score 0.989. The URL is extracted for the user, not visited." |
| 0:50–1:20 | Click **Bank OTP (legit)**. | "A legitimate one-time code. Band: looks ordinary; score 0.271. This is the case that matters: a bank alert must not be auto-blocked." |
| 1:20–1:45 | Click **Dinner plan**. | "Ordinary chat. Band: needs review — the conformal set is empty, so the app asks for review instead of guessing. Score 0.911." |
| 1:45–2:05 | Click **Invoice**. | "A legitimate invoice message. Band: likely scam. That is the known failure: legitimate transactional text reads as spammy. We show it rather than hide it." |
| 2:05–2:25 | Click **Hinglish KYC**. | "A Hinglish KYC lure. Band: likely scam; score 1.000. The pool is mostly English, so this is a demonstration, not a multilingual claim." |
| 2:25–2:45 | Point at the footer recipe line. | "Frozen: pool train, Platt and conformal on cal_a only. Scoring takes 2.84 ms on CPU; the artifact is 3.8 MB." |
| 2:45–3:00 | Stop. | "The ranking transfers better than the threshold. The demo is deliberately not an auto-blocker." |

## Likely questions

- **Why no accuracy?** The pool test set is 85.29% positive on 6,090
  rows, so accuracy would mostly restate the base rate. We report recall,
  FPR and PR-AUC per source instead.
- **Why is FPR about 0.5 on gold?** Gold is
  synthetic and has only 35 ham rows, many of them legitimate
  bank/service messages. The pool has almost no real bank-alert ham, so the
  model never learned that pattern. The interval is wide and the absolute
  FPR is not a field estimate.
- **Why is mmBERT not hosted?** It needs 1610 MB peak RSS and
  1205.9 MB on disk, against
  227 MB and 3.8 MB
  for svm, and it is optional behind `SMISHLENS_ENABLE_MMBERT=1`.
- **Why is the artifact private?** The bundle is a TF-IDF vocabulary plus
  SVC weights derived from pool text. MOZ, ExAIS and Smishtank have
  unverified or missing text licences, so redistributing the bundle is not
  clearly safe. It stays on the `private` remote; the public repository is
  code and text-free data only.
- **Does it generalise to new sources?** Ranking often does; the frozen
  threshold does not. `moz` is the counter-example where even ranking is
  near chance.
- **Was gold used for tuning?** No. Gold was scored once, after the
  recipe was frozen.

## Numbers used here

All values are read from `reports/results/*.json`:
`e3_pool.json`, `e6_gold_v2_bootstrap.json`, `e17_latency.json`,
`demo_bands.json` and `deploy_env.json` (Python 3.12.14).
`demo_bands.json` is written by `make demo-bands` from the frozen bundle;
it holds label, band, conformal set and score only, never message text.
