# GB Battery Co-Optimisation Terminal — developer entry points.
# Works on POSIX shells. On Windows use Git Bash, WSL, or run the underlying
# commands directly (see docs/deployment.md).

PY ?= python
VENV ?= .venv
ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts
else
	BIN := $(VENV)/bin
endif
PYTHON := $(BIN)/python
PIP := $(PYTHON) -m pip

.DEFAULT_GOAL := help

.PHONY: help install install-frontend ingest-demo test lint typecheck run run-frontend backtest freeze-sample docker-up clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Create venv and install the backend (editable) with dev extras
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e "backend[dev]"

install-frontend: ## Install frontend dependencies
	cd frontend && npm install

ingest-demo: ## Build/refresh the frozen public-data demo sample (works offline)
	$(PYTHON) -m gb_battery.cli ingest-demo

test: ## Run the backend test suite
	cd backend && ../$(BIN)/python -m pytest -q

lint: ## Ruff lint the backend
	cd backend && ../$(BIN)/python -m ruff check gb_battery tests

typecheck: ## mypy type-check the backend
	cd backend && ../$(BIN)/python -m mypy gb_battery

run: ## Run the FastAPI backend (http://localhost:8000)
	$(PYTHON) -m uvicorn gb_battery.api.main:app --app-dir backend --reload --port 8000

run-frontend: ## Run the Next.js dev server (http://localhost:3000)
	cd frontend && npm run dev

backtest: ## Run a demo chronological backtest and print a summary
	$(PYTHON) -m gb_battery.cli backtest

freeze-sample: ## Regenerate the frozen synthetic sample parquet files
	$(PYTHON) -m gb_battery.cli freeze-sample

docker-up: ## Build and start the full stack with Docker Compose
	docker compose up --build

clean: ## Remove caches and build artefacts
	rm -rf backend/.pytest_cache backend/.mypy_cache backend/.ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
