.PHONY: data features validate-rng test clean

PYTHON ?= python3
export PYTHONPATH := .

data:
	$(PYTHON) -m synth.generate --seed seed.yaml --out data

features: data
	$(PYTHON) scripts/run_sql_features.py --data data --sql sql --out data

validate-rng:
	$(PYTHON) scripts/check_rng.py --seed seed.yaml

test:
	$(PYTHON) -m pytest -q tests/

clean:
	rm -f data/*.csv data/*.parquet
	@# keep data/.gitkeep
