# smishlens — Tier 1 (UCI anchor) build, test and demo targets.
#
# `setup` and `train` never touch the frozen test split: final_eval.py is a
# one-shot script that refuses to overwrite reports/results/test_metrics.json.

PY     := .venv/bin/python
PIP    := .venv/bin/pip
export PYTHONPATH := src

.PHONY: setup test lint train app serve eda data summary docs diag e17 clean

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	$(PY) data/download_data.py --only uci
	$(PY) data/build_uci_anchor.py
	@test -f artifacts/model.joblib || $(PY) -m smishlens.train

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

train:
	$(PY) -m smishlens.train

app:
	.venv/bin/streamlit run app/app.py

serve:
	$(PY) -m uvicorn serve.api:app --reload

eda:
	$(PY) -m smishlens.eda

# Modern pool: raw sources are git-ignored, so this is the only way to rebuild
# data/processed/modern/{train,cal,test}.csv locally.
data:
	$(PY) data/download_data.py
	$(PY) data/build_modern_pool.py

# Regenerate docs/RESULTS_SUMMARY.md, README.md, MODEL_CARD.md and
# docs/SLIDES_OUTLINE.md from reports/results/*.json. No hand-typed numbers:
# the scripts are the only source of every number in the documents.
summary: docs
docs:
	$(PY) -m smishlens.results_summary
	$(PY) -m smishlens.docs_gen

# Post-hoc diagnostics (svm only; exploratory, no arm adopted).
diag:
	$(PY) -m smishlens.diagnostics
	$(PY) -m smishlens.diag_evasion

# E17 latency/footprint; writes reports/results/e17_latency.json.
e17:
	$(PY) -m smishlens.e17_latency

clean:
	rm -rf .pytest_cache .ruff_cache
