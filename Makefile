# smishlens — Tier 1 (UCI anchor) build, test and demo targets.
#
# `setup` and `train` never touch the frozen test split: final_eval.py is a
# one-shot script that refuses to overwrite reports/results/test_metrics.json.

PY     := .venv/bin/python
PIP    := .venv/bin/pip
export PYTHONPATH := src

.PHONY: setup test lint train app serve eda data summary clean

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

# Regenerate docs/RESULTS_SUMMARY.md from reports/results/*.json. No hand-typed
# numbers: the script is the only source of the document.
summary:
	$(PY) -m smishlens.results_summary

clean:
	rm -rf .pytest_cache .ruff_cache
