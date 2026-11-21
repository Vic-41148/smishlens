# HANDOFF — smishlens, end of 2026-11-21

Text-free note. No SMS text, no gold messages, no raw rows.

## State

- Branch `main`, last commit before this window: `80b563d`
  (`docs(results): record the ExAIS rebuild and Tier-2 rerun`). This window
  adds the mmBERT wiring, its E3/E4 results, and the 58-row gold v2 holdout.
- Frozen modern pool hash **`0bb9f4c85b6760d6`**, 30,447 rows,
  prevalence **0.8527** (25,963 positive / 4,484 ham).
- Per source: imc25 24,462 (3 ham); mishra 599 (367 ham); moz 2,330 (1,784 ham);
  smishtank 726 (0 ham); exais 2,330 (all ham).
- Splits: train 19,487 / cal 4,870 / test 6,090.
- LOSO folds: `mishra` and `moz` both-class; `imc25` and `smishtank`
  recall-only; `exais` FPR-only. Both-class macro is **n = 2**.
- `ruff check .` clean; full suite **126 passed**.
- Gold first pass is `reports/results/e6_gold.json`, mtime 2026-11-21
  14:15:53, `scored_once: true`. It was **not** re-run in this window.
- Gold v2 holdout is `data/gold_holdout_v2.csv`: 58 synthetic rows, written by
  Claude on the user's behalf, frozen before any model scored it. It is
  committed separately and has not been scored yet.

## mmBERT (G3, E3, E4)

- G3 pass on the 6 GB card: base 3.39 GiB / 0.277 s per step; small 1.73 GiB /
  0.129 s per step (2,000 rows, 1 epoch, `max_len=128`, `bs=16`, frozen
  embeddings).
- E3 headline, mmBERT-base, test scored once: macro **0.8890 recall / 0.0056
  FPR / 0.9814 PR-AUC** (classical svm: 0.8098 / 0.0097 / 0.9773). `T=2.543`,
  `tau=0.9610`, both from `cal_a`; conformal from `cal_b`.
- E4 LOSO, mmBERT-small, five folds: macro **0.6041 / 0.2178 / 0.5701**
  (classical svm: 0.4029 / 0.1678 / 0.5473). ExAIS FPR 0.2948 (svm 0.3811).
  Both models are weak under source shift; the gain is concentrated in moz and
  imc25.
- Base/small assignment is role/compute-based (G3 memory and throughput plus the
  roadmap compute plan), fixed before any test or LOSO number. No train/cal
  performance rule exists.
- Checkpoints under `artifacts/mmbert/` are git-ignored; hashes are in
  `reports/verification_log.md`.

## Committed vs local-only

Committed (after this window):

- Code: `src/smishlens/pool.py`, `pool_experiments.py`, `loso.py`, `probes.py`,
  `evaluate.py`; build/audit scripts in `data/`.
- Text-free pool artifacts: `split_manifest.csv`, `loso_folds.json`,
  `data_audit.json`, `label_overrides.json`, `dedup_exclusions.json`.
- Tests, results JSONs (including `reports/results/pre_exais/`), data cards,
  `reports/verification_log.md`, this file.
- Gold holdouts: `data/gold_holdout.csv` (133 rows, v1) and
  `data/gold_holdout_v2.csv` (58 rows, v2). Both fully synthetic.

Local-only / git-ignored:

- `data/raw/**` (PII and licence-restricted).
- `data/processed/modern/{train,cal,test}.csv` (processed text).
- `data/processed/review/**` (`review_sample.csv`).
- `artifacts/`, `models/`, checkpoints.
- Session exports.

Rebuild with `make data`; it loads `label_overrides.json` and
`dedup_exclusions.json` from `data/processed/modern/`.

## Open decisions

- The 58-row gold v2 holdout exists and is frozen. The next step is **one**
  second gold pass writing `reports/results/e6_gold_v2.json`; `e6_gold.json`
  stays untouched. Gold v1 was scored before the ExAIS change, so it is no
  longer pristine for that decision; v2 is.
- Do not widen `dedup_exclusions.json` from another source without a new
  pairwise audit. The current list is audited, not exhaustive.
- Keep the both-class macro at n = 2. ExAIS is FPR-only; never fold it into a
  both-class macro.

## Next work

1. Phase 1 (current): one gold v2 pass, E13 calibration, E12 PPV, E14 conformal,
   RQ1 gap write-up.
2. Phase 2: evasion E7–E10 and the Hinglish slice E11.
3. Phase 3: serving, packaging and the live URL.
4. Re-run E3/E4 only if the pool changes; otherwise reuse these results.

## Caveats

- **ExAIS held-out FPR is 0.381** (E4 svm). In-training E3 FPR is 0.0322. The
  gap is the real cost of the ham-only source.
- **ExAIS-vs-rest format AUC is 0.947** (E5). Format is close to a source
  fingerprint; never quote a pooled score.
- **UCI-only still beats the pool on macro PR-AUC** in E6: 0.669 vs 0.547
  (unweighted) and 0.583 (source-weighted). Pool arms cut ExAIS FPR
  (0.376 → 0.274 weighted) but lose 0.49–0.59 recall on both-class sources.
- **Unaudited near-dups may remain** in the 0.86–0.90 cosine band. The 8-row
  list only covers the audited Mishra-vs-UCI set.
- E4 moved a lot against the pre-ExAIS run: MOZ recall 0.026 → 0.319, Mishra
  recall 0.989 → 0.487, macro PR-AUC 0.580 → 0.547. That is pool composition
  (ExAIS ham in four training folds), not a code regression.
- The old Tier-2 section in `verification_log.md` is marked superseded.

## Commands that matter

```bash
# rebuild the frozen pool (loads overrides + exclusions from the out dir)
make data

# explicit build
PYTHONPATH=src .venv/bin/python data/build_modern_pool.py

# Tier-2 rerun without gold
PYTHONPATH=src .venv/bin/python -m smishlens.pool_experiments \
  --only E5,SEL,E3,E4,E6 --skip-gold

# checks
PYTHONPATH=src .venv/bin/python -m ruff check .
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## Gotchas

- **E6 parallelism:** use processes over folds, not threads over
  (model, fold). Threaded liblinear shares RNG state; two threaded runs differed
  (~1e-6 in PR-AUC/tau). The default `--n-jobs 6` is the process path.
- **E4 is slow:** about 20 minutes on this machine with the new pool (per-CV
  vectoriser, 5 folds × 5 models). The old 9-minute estimate is stale. The
  mmBERT-small E4 arm is five fine-tunes and takes ~35–40 minutes on the GPU.
- **Exclusion list:** applied at the UCI-overlap stage, before exact/near
  counts. Every id must be present or the build raises (stale list). The audit
  records `n`, `applied`, `reason`, `row_ids`.
- **Gold output path:** `score_gold` writes `e6_gold.json` directly. The second
  pass must point at `e6_gold_v2.json`; never overwrite the first pass.
- **Hardcoded-36 test:** recomputes the pre-UCI frame from raw and skips when
  raw/processed data is absent. It takes ~30–60 s locally.
- **`data/apply_audit.py --apply`** also loads `dedup_exclusions.json`, so a
  label-audit rebuild keeps the exclusion list.
