# SMS-2026: Modern Smishing & Spam Detection · FINAL ROADMAP (v4)

> **Capstone Project 2 (NLP)** · written 21 Nov 2026 · goal: pass the capstone PDF **and** end with a resume/research-grade project that is relevant in 2026. **v4 supersedes v3** (UCI-anchored robustness plan). What changed and why is in §0.2. Save as `ROADMAP.md` in the repo root. It is your design doc, sprint plan, literature notes, demo script and viva prep in one place.

---

## 0. How to read this file

### 0.1 Trust legend (every claim carries one of these)

|Tag|Meaning|
|---|---|
|✅|Checked by actually running code. The snippets tagged ✅ in §7 were executed in a sandbox before they went in here|
|📄|Paper-, model-card- or vendor-reported number. **Re-verify before quoting on a resume or slide**|
|🎯|A target, estimate or hypothesis. **Not a fact until your own experiment says so**|
|🔍|Needs a lookup (licence, column names, API, regulation text) before you rely on it|

**Hard rule:** every number on your resume, README or slides must be traceable to a committed file in `reports/results/`. Resume bullets in §17 use `[X]` placeholders on purpose.

### 0.2 What changed from v3, and why

|Area|v3|v4|Why|
|---|---|---|---|
|Data|UCI (2011) is the training set; modern data is an OOD test only|**Multi-source modern pool** (IMC'25, Smishtank, Mishra & Soni, MOZ-Smishing, ExAIS, UCI/NUS ham), tagged by source. UCI stays as the benchmark and the capstone anchor|You asked for modern, resume-grade data. Public classifiers collapse off UCI 📄 (§2)|
|Headline experiment|Evasion and PPV|**Leave-one-source-out (LOSO)** generalisation, plus a **source-shortcut probe**|With multiple sources, the main risk is that the model learns _which source a message came from_, not _what a scam looks like_|
|Transformer|DistilBERT|**mmBERT-base / mmBERT-small** (multilingual ModernBERT-style encoder, MIT licence). DistilBERT demoted to an optional English baseline|Multilingual, long-context, current (ICML 2026) 📄. Fits a 6 GB card with the tricks in §5.3 🎯|
|Product layer|Calibration + conformal|Adds a **policy layer** (deliver / junk / quarantine), an **explainer** (token attribution), integration **surfaces** (§10) and a **5-minute demo script** (§11)|"How do I actually use this?" is part of the project|
|LLM|"Small-LLM comparison row" in Tier 3|An **optional, non-deciding explainer helper**: local Qwen3-4B by default, API as a switchable backend, **never in the decision path**|Zero-shot LLMs were unreliable in SpaLLM-Guard 📄 and are attackable 📄. They are fine for phrasing an explanation|
|Link-less scams|Not covered|A **no-URL slice** and a URL-masked ablation|One 2026 vendor report says ~4 in 5 reported threats carried no link 📄. A URL-feature model would miss them|
|External baseline|Pretrained HF spam models banned as "yours"|Still banned as _your_ model, but **OpenTextShield** is run as a clearly-labelled **external baseline** on your held-out sources|Gives the comparison table a real-world reference row|
|Compute|"Colab T4 is enough"|Explicit **6 GB VRAM / 16 GB RAM budget** (§5.3), including the LLM helper|You run this on your own machine|

### 0.3 Bugs and wrong claims carried over from the old drafts (still fixed in the code below)

| #   | Old problem                                                                                                        | Fix                                                                                                      |
| --- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| 1   | `pd.read_csv(..., errors="replace")` raises `TypeError`                                                            | Use `encoding_errors="replace"` ✅                                                                        |
| 2   | String `"make_scorer(matthews_corrcoef)"` in the scoring dict                                                      | Pass the real `make_scorer(...)` object ✅                                                                |
| 3   | `CalibratedClassifierCV` saved at top level, `.named_steps` called on it, `load_model()` returned different shapes | Calibration is the **last Pipeline step**; `load_model()` always returns one `Bundle` ✅                  |
| 4   | "NFKC stops homoglyph attacks"                                                                                     | **It does not.** NFKC folds fullwidth letters but leaves Cyrillic `е`/`а`/`о`. Use a confusables map ✅   |
| 5   | Single marginal conformal threshold                                                                                | **Class-conditional (Mondrian)** conformal; spam is the minority class ✅                                 |
| 6   | ONNX-in-browser assumed to work for TF-IDF                                                                         | `skl2onnx` string ops may not run in ORT-Web 🔍. Test early; the safe route is a hand-rolled JSON scorer |
| 7   | Injection-block-rate as a headline metric                                                                          | Circular if you write both the regexes and the tests. Dropped                                            |

### 0.4 Open items that gate the plan (resolve these first, in order)

|#|Open item|How to resolve|Gate|
|---|---|---|---|
|G1|What columns does `final_dataset_output.csv` (IMC'25) have? Is the message text a column, or only screenshot references? Is there a date?|Run the inspection script in §21 and paste the output|Decides the ham-matching and time-split plan (§4.5)|
|G2|Which ham sources match the modern smish sources?|Audit MOZ-Smishing (has legit), ExAIS (has ham), Mishra & Soni (has ham, overlaps UCI), plus the HF leads in §4.2|Decides whether LOSO has both classes in each held-out source|
|G3|Does `mmBERT-base` fine-tune in 6 GB?|One-hour smoke test on Day 10 (§5.3)|Else use `mmBERT-small` as the main model|
|G4|Does mmBERT handle romanised Hinglish?|Test on the gold holdout (E11)|Decides whether Hinglish is a claim or a limitation|
|G5|Licences: MOZ-Smishing, COVA-X, Smishtank, Gemma 4|See §21|Decides what you may redistribute|

---

## 1. TL;DR

**The vanilla project ("TF-IDF + Naive Bayes on UCI + a Streamlit box") is the floor everyone submits.** In-distribution accuracy on UCI is saturated at about 98% 📄. The interesting question is what happens **off** UCI: on new scam templates, new countries, new languages, disguised characters and messages with no link.

**Working pitch (rename it, see §6.1):** _"A calibrated, explainable smishing/spam detector that is evaluated on sources it has never seen, hardened against character-level evasion, and honest about when it is unsure."_

### 1.1 Three tiers, each ends in something you can submit

|Tier|Days|What you get|Value|
|---|---|---|---|
|**1 · MVP**|≈7|UCI-only classical pipeline, evaluation, calibrated confidence, Streamlit app, README. **Meets every line of the capstone PDF**|Safe submission|
|**2 · SMS-2026 (the real target)**|+12|Modern multi-source dataset, LOSO and shortcut probe, mmBERT, evasion suite + normaliser, no-URL slice, calibration, conformal abstention, policy layer, explainer, API, CI gate, deploy, demo|**Resume-grade**|
|**3 · Stretch**|+7–10|Browser-only demo, Android share-sheet, Telegram bot / webhook, conversational (COVA-X) early detection, write-up. **Pick 1–2**|Bonus|

**Rule:** finish a tier completely (README + working GUI + git tag) before starting the next.

### 1.2 Decisions locked in

|Decision|Choice|Why|
|---|---|---|
|Task framing|**Binary headline:** `unwanted/malicious (spam ∪ smishing)` vs `ham`. **3-class** (ham / spam / smishing) only where labels exist|Matches the capstone's "Spam / Not Spam". The spam-vs-phishing boundary is the hardest part even for a deployed model 📄|
|Classical baseline|`word TF-IDF ∪ char_wb TF-IDF ∪ meta → LinearSVC(balanced) → Platt (sigmoid)`|Required by the PDF, tiny, interpretable, a real reference row|
|Main model|**mmBERT-base** (307M, 110M non-embedding) 📄; **mmBERT-small** (140M) as the CPU/deploy variant|Multilingual, modern, MIT licence 📄|
|External baseline|**OpenTextShield** (public multilingual BERT, MIT 📄), run on your held-out sources|Real-world reference. Never described as your model|
|Headline metrics|**Leave-one-source-out** recall/FPR/PR-AUC, FPR-constrained recall, PPV at 5/10/20% spam, evasion recall|A blocked OTP costs far more than a missed ad|
|Decision rule|Calibrated probability + FPR-budget threshold + class-conditional conformal → **deliver / junk / quarantine**|One brittle "AI says spam, drop it" vector is the failure mode operators warn about 📄|
|Explainer|Token attribution (linear coefficients for the SVM, **occlusion** for mmBERT). LLM phrasing is optional|Faithful to the real model|
|LLM helper|**Off by default.** Local Qwen3-4B-Instruct (Q4) if on; API (Claude Haiku 5.5) as a switchable backend; never decides|Privacy, zero cost, offline demo|
|Serving integrity|SHA-256 manifest check (T1/T2); ONNX / JSON scorer (T3)|Pickle is code execution on load|
|Surfaces|Streamlit demo + FastAPI (Docker) first. Browser-only, Android share-sheet and bot are Tier 3|Android/iOS SMS access is gated by platform rules (§10.2)|
|Deploy|Hugging Face Spaces or Streamlit Community Cloud|Live link at the top of the README|

### 1.3 Deliberately **not** doing (say this in the report, it reads as maturity)

An LLM as the classifier · RAG/vector DB · a LangChain agent · Kubernetes · SMOTE on sparse TF-IDF (run once at most, report "no gain") · presenting someone else's fine-tuned model as yours · a "spam bypass tool" packaging of the attack module · building the plan on **Super SMS** (§4.2: no confirmed public download) · scraping live URLs · training on LLM-generated data and then testing on it.

---

## 2. State of the field (Nov 2026), condensed

All numbers here are 📄 (paper, model card or vendor reported). Re-verify against the sources in §22 before quoting.

|Finding|Detail|Implication for you|
|---|---|---|
|UCI is saturated|Classical models ≈ 98% accuracy on UCI. A 2026 MDPI study: TF-IDF + linear SVM ≈ 98.15% accuracy but F1 ≈ 91.7|Another 0.2% on UCI is not a result. Dedup and split choices move F1 more|
|**Classifiers drop hard off UCI**|OpenTextShield's own model card (mBERT, ~180M params, 3 classes) lists a **99.5% block rate on UCI**, **99.3% on Mishra & Soni**, but only **72.4% block rate on the IMC 2025 smishing set (8,007 msgs)** and **45.7% phishing recall** there. The card says spam-vs-phishing is the hardest boundary (my copy of the table headers was partly garbled: 🔍 verify which column is which)|This gap is the story of the project. Report cross-source numbers first, UCI second|
|Encoding attacks are the gap|"Johnny Still Receives Spam SMS" (Sept 2026): RF, SVM and LightGBM dropped **below 20% accuracy** under all attacks; BERT resisted visible edits but fell to **41.3% on homoglyphs**; best open model FNR ≈ 9.4%|A normaliser is the cheap baseline (RQ3). Report whatever you find|
|LLMs are not a free fix|SpaLLM-Guard: zero-shot is unreliable, few-shot varies, fine-tuning helps, **small chat LLMs are not robust to SMS spam**. Other work shows LLM spam filters are susceptible to adversarial and poisoning attacks. A URL-masked fraud benchmark had frontier LLM agents at ~42–51% against 47.5% for "always fraud"|LLM stays out of the decision path|
|Scams went link-less|SmishAlert Q2 2026 (a **vendor** report, not peer-reviewed): about 4 in 5 reported threats carried no link and opened with a conversation|Do not lean on URL features. Test a no-URL slice (E10)|
|Prevalence kills precision|With TPR 98.66% and FPR 0.82%, precision falls from ≈99.2% (balanced) to ≈86.4% at 5% spam ✅ (arithmetic checked)|Report PPV at realistic prevalence (RQ4)|
|Real deployments combine signals|PAiCore/OpenTextShield (GSMA member case study): content classifier at the SMSC with programmable policy; "a single-vector firewall is brittle"; private on-prem inference; vendor claims <1 s per message and ~100 msg/s on a T4|Your policy layer (deliver/junk/quarantine) copies this pattern. Publish **your own** latency numbers|
|Fresh data exists but is messy|IMC'25: 64.5k+ smishing reports/attachments from five public forums, labelled by scam type and lure, CC BY 4.0 (repo confirmed by you). Smishtank ≈1.1k validated smishes. MOZ-Smishing (Mozambique, 552 smish / 2,009 legit). COVA / COVA-X synthetic multi-turn conversations (3,201 → 10,985)|§4 builds a pool from these|
|Platform rules limit "real filters"|iOS Message Filter extensions see only unknown senders and cannot use the network themselves. Google Play restricts SMS permissions to default handlers, with a case-by-case exception for spam detection|Ship as a checker/API first (§10)|
|India context|TRAI issued AI/ML spam-flagging directions in 2026 and industry flagged false positives blocking critical alerts 🔍|Frames the FPR-first operating point. Most TRAI measures concern calls/numbers and operator-side signals; don't claim they regulate content classifiers|

**Where your system sits:** operators use behavioural signals (velocity, SIM churn, headers/DLT templates). A content classifier like yours is the complementary on-device / gateway layer. Put that on a slide.

---

## 3. Thesis and research questions

**Thesis:** _A filter that scores 98% on one dataset is not one you can trust. Measure how it behaves on sources, scam styles and disguises it has never seen, close part of the gap cheaply, and make it say "I'm not sure" instead of guessing._

|RQ|Question|Experiment|Expectation (all 🎯)|
|---|---|---|---|
|**RQ1**|How much does detection drop when a whole **source** is held out (LOSO), and does training on a modern pool beat UCI-only?|E4, E6|Expect a large drop for UCI-only, smaller for the pool. Size unknown|
|**RQ2**|Is the model learning scam content, or the **source/format** it came from?|E5|A format-only probe with high AUC is a red flag|
|**RQ3**|Does normalisation (NFKC + strip format chars + confusables) recover detection under evasion **without retraining**?|E7 vs E8|Should help encoding attacks, **not** leetspeak or spacing. Report truthfully|
|**RQ4**|What do precision and alert volume look like at 5/10/20% spam prevalence?|E12|Arithmetic is deterministic ✅|
|**RQ5**|Is the confidence trustworthy, and does conformal abstention hold up **under source shift** (where its assumption breaks)?|E13, E14|Coverage likely degrades under LOSO. Show how much|
|**RQ6**|How much does performance depend on URLs?|E10|URL-masked vs link-less slice|
|**RQ7** (T3)|Do multilingual encoders help on Hinglish / non-English slices?|E11|Unknown|

Claim a **rigorous, reproducible, multi-source evaluation with honest uncertainty**, not novelty. That is truthful and still strong.

---

## 4. Data

### 4.1 Pools

|Pool|What|Purpose|Rule|
|---|---|---|---|
|**A: anchor**|UCI SMS Spam Collection v.1 (id 228): 5,574 msgs, 4,827 ham / 747 spam (≈13.4% spam). HF `ucirvine/sms_spam` mirror; Kaggle CSV needs `latin-1`|Satisfies the PDF, comparable with the literature, extra ham|Dedup **before** splitting. Licence CC BY 4.0 per the UCI page|
|**M: modern pool**|IMC'25, Smishtank, Mishra & Soni, MOZ-Smishing, ExAIS, NUS ham (§4.2)|Main training data and the LOSO sources|Tag **every row with its source**. Overlap-check before merging|
|**C: gold holdout**|100–150 messages **you write**: modern parcel / UPI / KYC / bill / job-offer / "wrong number" scams (including **link-less** ones and **Hinglish**), plus hard-negative ham (real-looking OTPs, bank alerts, delivery updates)|"Test on new SMS" requirement, E11, E16, live demo|Written **before** you see test errors. Invented or defanged, never real personal SMS|
|**D: synthetic**|COVA / COVA-X conversations, Mendeley "balanced" LLM-generated set, your own LLM-made variants|Augmentation and the Tier 3 conversational task|`synthetic=True`, ≤30% of train, **never** in a headline test set|

**Unified schema** (every loader returns this): `text, y, y3, source, lang, date, has_url, scam_type, synthetic`. Use `NaN` where a source doesn't provide a field. `y` is binary (1 = spam or smishing). `y3` is `ham/spam/smish` where known.

### 4.2 Source catalogue (what is real, what is a trap)

|Source|Size|Contains|Use here|Check before use|
|---|---|---|---|---|
|**UCI SMS Spam** (UK/Singapore, ~2011)|5,574|ham + spam|Anchor, benchmark, ham|~400 exact duplicates 📄. Includes 3,375 NUS ham, so overlaps NUS|
|**IMC'25 smishing** (`reportsmishing/Smishing-Dataset-IMC25`)|Paper: 64.5k+ reports/attachments from five public forums. Repo: `dataset/final_dataset_output.csv` (labelled) + `time_day.csv`|**Smishing only.** Scam types (7), lure principles, languages|Main modern positives. Possible time split|**CC BY 4.0 ✅** (repo README). Columns, whether message text is a column, and date coverage 🔍 (G1). Text may be extracted from screenshots → shortcut risk (§4.3)|
|**Smishtank**|≈1.1k validated|Smishing only|Extra positives, recall-only OOD|Terms of use 🔍|
|**Mishra & Soni** (Mendeley `f45bkkt8pr`)|5,971: 4,844 ham / 489 spam / 638 smish|3-class|3-class labels, ham|**≈4,753 rows overlap UCI** 📄. Remove exact **and** near-duplicates first|
|**MOZ-Smishing** (HF `MOZNLP/MOZ-Smishing`)|552 smish / 2,009 legit|Mobile-money smishing, Mozambique, crowd-sourced|Held-out **region** with both classes|Licence stated as "open" 🔍. Languages 🔍|
|**ExAIS** (GitHub `AbayomiAlli/SMS-Spam-Dataset`)|5,240: 2,350 spam / 2,890 ham, 20 users, Nigeria|African-English, PII masked|Held-out source with both classes|2015 data, older style|
|**NUS SMS Corpus** (`WING-NUS/nus-sms-corpus`)|67,093 (2015)|Mostly legitimate chat|Extra ham only|Authors ask for a note when used. Licence 🔍. Overlaps UCI's NUS ham|
|**COVA / COVA-X**|3,201 → 10,985 synthetic multi-turn conversations, 8 elder-targeted scam types (COVA is CC BY 4.0)|Conversations, per-conversation labels|Tier 3 conversational early detection|Synthetic. COVA-X download and licence 🔍|
|**Mendeley "balanced"**|10,191 LLM-generated (3,397 per class)|Synthetic, derived from Mishra & Soni|Train augmentation only|Inherits Mishra's UCI overlap|
|**HF leads** (`CloveAI/india-spam-sms`, `Ngadou/Spam_SMS`, `daviiiidcpp1/sms-spam-combined`, `electricsheepafrica/africa-smishing-sms-phishing`, `ealvaradob/phishing-dataset`)|?|Named on a model card as audited by someone else|**Unverified leads.** `india-spam-sms` is the one worth checking for India relevance|Audit size, labels, licence, overlap. Combined sets inherit UCI overlap|
|**WSPR Lab** (`wspr-ncsu/sms-phishing`)|?|Phishing campaigns|Optional extra OOD|Licence and format 🔍|

**Do not use / do not build on:**

- **Super SMS dataset.** Papers describe it (67,010 messages; 40,837 benign / 26,181 spam; aggregates UCI, NUS and Spamhunter plus Twitter, ScamWatch, UK Action Fraud), but the original paper only _promised_ a public release, its reported size differs between papers (153,551 vs 67,010 vs 67,018), and I found **no confirmed public download**. It also contains UCI rows. If a real link appears, add it as a source and treat it like any other.
- **The GitHub "Combined Labeled Smishing Dataset":** it relabels messages by the presence of smishing keywords, so a keyword model is graded against its own shortcut.
- Unvetted aggregator sets (e.g. a "149k multilingual" community post). Sample and hand-audit 100 labels first, or skip.

### 4.3 The central trap: the model can learn the _source_, not the scam

If smishing positives come from 2024 forum screenshots and ham comes from 2011 UCI, a classifier can score 99% by detecting **era, punctuation style, OCR artefacts or casing**. Controls, all of which are experiments (E5):

1. **Match ham to each positive source wherever a source has its own ham** (MOZ-Smishing legit, ExAIS ham, Mishra & Soni ham).
2. **Normalise formatting** before modelling: Unicode (curly quotes → straight), whitespace, case, zero-width characters.
3. **Format-only probe** (§7.9): train a model that sees _only_ formatting features (length, punctuation, quote style, whitespace, non-ASCII ratio). Compare its AUC with the content model's. A format-only AUC that is close to the content model's is a red flag. It is **not** pass/fail: even clean data shows some AUC from length alone, so also report the AUC of length alone, and report it **within** each source.
4. **LOSO only on sources with both classes.** Positives-only sources (IMC'25, Smishtank) are **recall-only** tests.
5. **Hand-audit 100 labels per source** before training. Log disagreements.

### 4.4 UCI quirks to handle explicitly (each is a report bullet)

1. **~400 exact duplicates** (≈5,169 unique) 📄 → verify in your notebook, dedup before splitting, and quantify the leakage inflation by reporting with and without dedup.
2. **Label conflicts:** identical text with both labels → drop.
3. **Encoding:** Kaggle `spam.csv` is `latin-1` with junk columns `Unnamed: 2..4`. The raw UCI file is tab-separated with stray quotes → `quoting=csv.QUOTE_NONE`.
4. **Length shortcut:** spam averages ~139 chars vs ~71 for ham 📄, and URLs appear in only ~1.9% of messages 📄. Check performance on **short spam** to prove it learned content.
5. **Limitations for the slide:** English only, UK/Singapore, ~2011, no modern smishing.

### 4.5 Data rules (these protect you from the mistakes reviewers look for)

1. Dedup first, then split. Normalise-then-dedup (lowercase, collapse whitespace) catches near-exact copies. Run the **cross-source near-duplicate check** (§7.9) between every pair of sources.
2. Fit every vectoriser/scaler **inside** a Pipeline on train only.
3. Never merge a second dataset into training without an overlap check (exact + normalised + char-n-gram similarity).
4. Keep held-out sources and the gold holdout untouched until the final evaluation.
5. `DATA_CARD.md` per source: origin, access date, licence, size, balance, languages, known biases, what you removed.
6. If a licence forbids redistribution, commit a download script, not the data.
7. No real personal SMS in the repo, ever (including your own). Defang URLs and numbers in every published example.
8. LLM-generated rows are tagged `synthetic=True` and never appear in a test set.

### 4.6 Splits (freeze on Day 8, never change)

```
per-source loaders → unified schema → drop label conflicts → exact dedup → near-dup removal across sources
  ├── UCI ANCHOR (capstone):   train 64% / cal 16% / test 20%, stratified, seed 42   (v3 split, unchanged)
  ├── MODERN DEV POOL (all modern sources):  64 / 16 / 20, stratified by (source, y)
  │        train → fitting, CV tuning, OOF threshold        cal → conformal + temperature scaling ONLY
  │        test  → touched exactly once
  ├── LOSO FOLDS: hold out ONE source that has both classes; train on the rest. Positives-only sources → recall-only
  ├── TIME SPLIT (only if IMC'25 has usable dates 🔍): train on earlier, test on later
  └── gold_holdout.csv (Pool C): never used for tuning
```

Calibration and conformal data come from the **in-pool** cal split. Under LOSO the held-out source is _not_ exchangeable with it, so coverage can degrade. That is exactly RQ5. Commit `train/cal/test.csv`, the fold definitions, the seed and a dataset hash.

---

## 5. Models and compute

### 5.1 Model line-up

|Model|Role|Notes|
|---|---|---|
|Dummy (always-ham, stratified)|Floor|Shows why accuracy is meaningless at 87/13|
|MNB, ComplementNB|Classical rows|Overconfident, so calibrate|
|LogReg, **LinearSVC (+Platt)**|Classical champion|Required baseline, 1–4 MB|
|fastText (optional)|Tiny robustness row|Often brittle under imperceptible attacks 📄|
|**mmBERT-small** (140M, 42M non-embedding) 📄|CPU/deploy variant|ONNX int8 for browser/CPU (Tier 3)|
|**mmBERT-base** (307M, 110M non-embedding) 📄|**Main model**|MIT licence, 8,192 context, 1,800+ languages, ICML 2026 📄. Uses the Gemma 2 tokenizer 📄|
|DistilBERT-base (optional)|English baseline|Cheap extra row if time allows|
|**OpenTextShield** (external)|Reference|Public mBERT, 3 classes, MIT 📄. Map its spam+phishing → 1. Report on **your** held-out sources, labelled as external|

### 5.2 Task framing

- **Headline:** binary (`spam ∪ smishing` = 1). It matches the capstone and sidesteps the hardest boundary.
- **Secondary:** 3-class on sources that have it (Mishra & Soni and the Mendeley set). Report the confusion between spam and smishing explicitly.

### 5.3 Compute budget (your machine: RTX 3050 6 GB VRAM, 16 GB RAM, ~2 GB used by someone else)

|Resource|Budget|What goes there|
|---|---|---|
|VRAM 6 GB|Training: all of it. Serving: ~4 GB for the LLM helper, ~2 GB left for display and mmBERT|**Never** train and serve the LLM at the same time|
|RAM ~14 GB|App stack 1–2 GB, int8 mmBERT on CPU ~1 GB, browser and OS the rest|Do not push LLM layers into RAM unless you must: CPU layers are much slower|

**mmBERT-base fine-tune memory (my estimate 🎯, test on Day 10 = gate G3).** Total 307M parameters, of which about 197M are the token-embedding matrix (the large multilingual vocabulary) and 110M are transformer weights.

- Naïve full fine-tune with AdamW in fp32 costs roughly 16 bytes per parameter ≈ **4.9 GB before activations**: too tight on 6 GB.
- **Freeze the embedding matrix** → ~110M trainable: weights 1.2 GB + gradients 0.44 GB + Adam 0.88 GB ≈ **2.5 GB + activations**. This fits.
- Settings to try: `max_len 128`, batch 16 (or 8 × gradient-accumulation 2), bf16/fp16 autocast (the 3050 is Ampere), gradient checkpointing if you hit OOM.
- **Fallbacks, in order:** `mmBERT-small` as the main model → Colab T4 for the base model.

### 5.4 The optional LLM helper (explainer only)

The encoder decides. The LLM, if turned on, only phrases one plain sentence for **uncertain/quarantined** messages. It cannot change a verdict or action.

|Option|Size / cost|Fit|Verdict|
|---|---|---|---|
|**Qwen3-4B-Instruct-2507, Q4_K_M** (Apache 2.0) 📄|about 3 GB|Fully in 6 GB VRAM. Expect roughly 20–30 tok/s 🎯|**Default.** `ollama pull qwen3:4b-instruct`|
|Qwen3-8B, Q4_K_M 📄|5.03 GB weights|Too tight once the KV cache is added; needs partial GPU offload (`-ngl ~30` 🎯)|Upgrade path only|
|Gemma 4 E4B (GGUF exists: `ggml-org/gemma-4-E4B-it-GGUF`)|size unverified|Probably|Rival in the same 20-message test. Licence 🔍|
|**Claude Haiku 5.5 (API)**, id `claude-haiku-5-5` 📄|$0.10 in / $0.50 out per million tokens (≤100K prompt); batch $0.05 / $0.25|n/a|Switchable backend. Costs ≈ $0.00007 per explanation (300 in + 80 out tokens, my arithmetic 🎯). **Sends text off-device**, which breaks the privacy story|
|Qwen3-30B-A3B at ~2.95-bit|~11 GB|Hybrid GPU/RAM, one author's tool claims ~16 tok/s|Skip: heavy compression, thin RAM margin, overkill for one sentence|

**Rules:**

1. **`EXPLAINER=off|local|api`**, default `off`.
2. The prompt contains **only** the label, the probability and up to 6 highlighted **single words**, sanitised (§7.13). **Never the raw SMS**: it is attacker-controlled text and LLM filters are attackable 📄.
3. The UI shows the LLM sentence in a labelled "explanation" slot. Verdict and action always come from the encoder and policy.
4. Cache explanations per message (instant live demo).
5. Before committing, run **20 messages (English + Hinglish)** through each candidate and record tokens/s, peak VRAM (`nvidia-smi`) and whether each sentence stays faithful to the highlighted words.
6. **A better use of an API LLM:** offline, as a second opinion when you hand-audit labels, and to generate adversarial/Hinglish test variants. Tag all of that `synthetic=True`.

---

## 6. Architecture and repo

### 6.1 Rename the project

`chhanni` was rejected. Shortlist: `sms-bouncer` (my pick: it decides who gets into your inbox), `kachra-filter`, `hamlet`, `spamlens`, `smishwall`. Search each on GitHub, Hugging Face **and** PyPI before committing. If the PyPI name is taken, put a suffix on the package, not the repo. The package is called `smsguard` below as a placeholder.

### 6.2 Pipeline

```
raw SMS ─► [normalise: NFKC · strip format chars · confusables] ─► [encoder: calibrated SVM / mmBERT]
        ─► [calibrated p] ─► [threshold τ from FPR budget] ─► [conformal set: auto vs abstain]
        ─► [policy: deliver / junk / quarantine] ─► [explain: token attribution] ─► (optional LLM sentence)
Offline harness: LOSO · shortcut probe · evasion · no-URL slice · PPV@prevalence · calibration · conformal · latency
```

A diagram of this flow is in `docs/sms2026_pipeline.svg` (delivered with this roadmap).

Design rules:

- `normalise()` is its own module so every experiment runs with it on and off (RQ3).
- **One core, thin UIs.** All logic in `src/`, imported by notebooks, the app, the API and the tests, so training and serving cannot drift.
- Custom transformers (`CleanText`, `MetaFeatures`) live in an importable module, or `joblib` cannot unpickle them elsewhere.
- The raw SMS is kept only in memory for the request. No content logging.

### 6.3 Repo structure

```
smsguard/
├── data/{raw,processed}/             # download scripts + DATA_CARD.md per source; processed splits committed
│   └── gold_holdout.csv
├── notebooks/ 01_eda · 02_sources_audit · 03_models · 04_mmbert · 05_loso_shortcut · 06_robustness · 07_calibration_conformal
├── src/smsguard/
│   ├── config.py   data.py   sources.py  dedup.py   normalise.py   features.py   models.py
│   ├── train.py    train_mmbert.py   evaluate.py   loso.py   probes.py   attacks.py   conformal.py
│   ├── calibrate.py   policy.py   explain.py   llm_explainer.py   predict.py   tracking.py
│   └── eval/gates.py
├── serve/api.py                      # FastAPI
├── app/app.py                        # Streamlit
├── artifacts/                        # model.joblib / mmbert/, metadata.json, MANIFEST.sha256
├── reports/{figures,results,tables}/ verification_log.md
├── docs/ sms2026_pipeline.svg   demo_script.md
├── tests/ test_normalise.py test_ppv.py test_attacks.py test_conformal.py test_policy.py test_loso.py
│          test_probe.py test_predict.py test_artifact.py test_explainer_prompt.py
├── eval/floors.json                  # metric floors for the CI gate
├── .github/workflows/ci.yml
├── Dockerfile  Makefile  requirements.txt  requirements-dev.txt  README.md  MODEL_CARD.md  DATA_CARD.md  LICENSE
```

### 6.4 Artifact contract (`artifacts/metadata.json`)

```json
{
  "model_name": "mmbert_base_temp_scaled",
  "model_version": "1.0.0",
  "created_utc": "…",
  "sklearn_version": "…", "torch_version": "…", "transformers_version": "…", "python_version": "…",
  "task": "binary: spam_or_smish vs ham",
  "datasets": {"sources": ["uci","imc25","smishtank","mishra","moz","exais"], "dataset_sha256_16": "…",
               "rows_dedup": "…", "train": "…", "cal": "…", "test": "…", "seed": 42},
  "normaliser": {"nfkc": true, "strip_format_chars": true, "confusables": true, "mask_tokens": true},
  "calibration": {"method": "temperature", "T": "…"},
  "threshold": {"value": "…", "policy": "max recall s.t. OOF FPR <= 1%"},
  "conformal": {"alpha": 0.05, "q_hat": {"0": "…", "1": "…"}, "n_cal_spam": "…", "type": "class-conditional"},
  "policy": {"junk": "p>=tau and set=={1}", "quarantine": "set != single class", "deliver": "otherwise"},
  "explainer": {"mode": "off", "llm": null},
  "metrics_file": "reports/results/test_metrics.json"
}
```

Never hardcode thresholds in the app. The app and API read this file.

### 6.5 Prediction contract (shared by every surface)

```python
@dataclass(frozen=True)
class Prediction:
    label: str              # "Spam/Smishing" | "Not Spam"
    is_spam: bool
    p_spam: float           # calibrated
    threshold: float
    conformal_set: list     # [0], [1], [0, 1] or []
    action: str             # "deliver" | "junk" | "quarantine"
    trigger_words: list     # [(word, score_drop), ...]
    normaliser_flags: dict  # {"confusables": 3, "format_chars": 1}
    explanation: str | None # LLM sentence, only if EXPLAINER != off
    latency_ms: float
    model_version: str
```

`load_model()` returns **one** `Bundle(model, meta)` everywhere. `predict_one` never raises on weird input.
