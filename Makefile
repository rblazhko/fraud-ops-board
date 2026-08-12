.PHONY: data features train validate-rng test board clean

PYTHON ?= python3
export PYTHONPATH := .

data:
	$(PYTHON) -m synth.generate --seed seed.yaml --out data

features: data
	$(PYTHON) scripts/run_sql_features.py --data data --sql sql --out data

train: features
	$(PYTHON) -m model.train --features data/features_wide.parquet \
		--out artifacts/model.joblib --metrics reports/metrics.json

validate-rng:
	$(PYTHON) scripts/check_rng.py --seed seed.yaml

test:
	$(PYTHON) -m pytest -q tests/

board:
	$(PYTHON) -m streamlit run board/app.py --server.headless true

clean:
	rm -f data/*.csv data/*.parquet
	rm -f artifacts/*.joblib reports/metrics.json reports/metrics.md
	@# keep data/.gitkeep artifacts/.gitkeep reports/.gitkeep
