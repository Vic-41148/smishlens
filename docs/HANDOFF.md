# HANDOFF — smishlens, end of 2026-11-21

Text-free note. No SMS text, no gold messages, no raw rows.

## State

- Branch `main`, last commit before this window: `6341d3c`
  (`data(gold): add the 133-row synthetic gold holdout`).
- Frozen modern pool hash **`0bb9f4c85b6760d6`**, 30,447 rows,
  prevalence **0.8527** (25,963 positive / 4,484 ham).
- Per source: imc25 24,462 (3 ham); mishra 599 (367 ham); moz 2,330 (1,784 ham);
  smishtank 726 (0 ham); exais 2,330 (all ham).
- Splits: train 19,487 / cal 4,870 / test 6,090.
- LOSO folds: `mishra` and `moz` both-class; `imc25` and `smishtank`
  recall-only; `exais` FPR-only. Both-class macro is **n = 2**.
- `ruff check .` clean; full suite **123 passed**.
- Gold first pass is `reports/results/e6_gold.json`, mtime 2026-11-21
  14:15:53, `scored_once: true`. It was **not** re-run in this window.

## Committed vs local-only

Committed (after this window):

- Code: `src/smishlens/pool.py`, `pool_experiments.py`, `loso.py`, `probes.py`,
  `evaluate.py`; build/audit scripts in `data/`.
- Text-free pool artifacts: `split_manifest.csv`, `loso_folds.json`,
  `data_audit.json`, `label_overrides.json`, `dedup_exclusions.json`.
- Tests, results JSONs (including `reports/results/pre_exais/`), data cards,
  `reports/verification_log.md`, this file.

Local-only / git-ignored:

- `data/raw/**` (PII and licence-restricted).
- `data/processed/modern/{train,cal,test}.csv` (processed text).
- `data/processed/review/**` (`review_sample.csv`).
- `artifacts/`, `models/`, checkpoints.
- Session exports.

Rebuild with `make data`; it loads `label_overrides.json` and
`dedup_exclusions.json` from `data/processed/modern/`.

## Open decisions

- The user writes about **30 fresh gold messages**. Then run **one second gold
  pass** and keep it next to the first. The first pass stays untouched; any
  second pass must write `e6_gold_v2.json`.
- Do not widen `dedup_exclusions.json` from another source without a new
  pairwise audit. The current list is audited, not exhaustive.
- Keep the both-class macro at n = 2. ExAIS is FPR-only; never fold it into a
  both-class macro.

## Next work

1. mmBERT on the frozen pool (small for LOSO, base for the headline/final fit;
   see the G3 memory note).
2. E7–E14 (evasion suite, held-out-family arms, URL-masked arm, error analysis,
   WSPR OOD, etc. — follow the roadmap order).
3. Re-run E3/E4 only if the pool changes; otherwise reuse these results.

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
  vectoriser, 5 folds × 5 models). The old 9-minute estimate is stale.
- **Exclusion list:** applied at the UCI-overlap stage, before exact/near
  counts. Every id must be present or the build raises (stale list). The audit
  records `n`, `applied`, `reason`, `row_ids`.
- **Gold output path:** `score_gold` writes `e6_gold.json` directly. For the
  second pass, change the path to `e6_gold_v2.json` (or move v1 aside first);
  do not overwrite the first pass.
- **Hardcoded-36 test:** recomputes the pre-UCI frame from raw and skips when
  raw/processed data is absent. It takes ~30–60 s locally.
- **`data/apply_audit.py --apply`** also loads `dedup_exclusions.json`, so a
  label-audit rebuild keeps the exclusion list.
