# AgriSentinel — common tasks.
# Every target is a thin wrapper around the same entry points the README documents;
# nothing here does work that the Python package cannot do on its own.

PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help install test pipeline stages site clean

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:         ## Create a virtualenv and install the package with dev extras
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

test:            ## Run the unit-test suite
	$(BIN)/pytest tests/ -q

pipeline:        ## Run every stage: collect -> clean -> statistics -> mine_rules -> train -> export
	$(BIN)/python scripts/run_pipeline.py -v

stages:          ## List the individual pipeline stages
	$(BIN)/python scripts/run_pipeline.py --list-stages

site:            ## Serve the dashboard locally
	$(BIN)/python scripts/serve_site.py

clean:           ## Remove regenerable artifacts (data/processed, models, reports, caches)
	rm -rf data/interim/* data/processed/* models/* reports/figures/*.png reports/metrics/*.json
	rm -f site/agrisentinel_data.js
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache *.egg-info
