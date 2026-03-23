# ============================================================
# High-Paying Jobs in the US — Reproducibility Makefile
# ============================================================
# Usage:
#   make              — show this help
#   make install      — create .venv and install all dependencies
#   make data         — regenerate cleaned dataset from raw sources
#   make model        — train XGBoost model (via scripts/train_model.py)
#   make test         — run full pytest suite
#   make coverage     — run tests with coverage report
#   make lint         — ruff check (fast linter)
#   make format       — ruff format (opinionated auto-formatter)
#   make type-check   — mypy static type checker
#   make dashboard    — launch Streamlit on http://localhost:8501
#   make api          — launch FastAPI on http://localhost:8000
#   make docker       — build and start both services with Docker Compose
#   make mlflow       — launch MLflow tracking UI on http://localhost:5000
#   make clean        — remove generated artefacts (models, cache, .pyc)
#   make clean-all    — clean + remove the virtual environment
# ============================================================

PYTHON    := python3
VENV      := .venv
PIP       := $(VENV)/bin/pip
PYTEST    := $(VENV)/bin/pytest
RUFF      := $(VENV)/bin/ruff
MYPY      := $(VENV)/bin/mypy
JUPYTER   := $(VENV)/bin/jupyter
STREAMLIT := $(VENV)/bin/streamlit
UVICORN   := $(VENV)/bin/uvicorn

# Detect OS for open-browser command
UNAME := $(shell uname -s)
ifeq ($(UNAME), Darwin)
  OPEN := open
else
  OPEN := xdg-open
endif

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  High-Paying Jobs in the US — available targets"
	@echo "  ------------------------------------------------"
	@echo "  install     Create .venv and install all dependencies"
	@echo "  data        Regenerate cleaned dataset from raw sources"
	@echo "  model       Train XGBoost model → models/xgb_salary_model.ubj"
	@echo "  test        Run full pytest suite (67 tests)"
	@echo "  coverage    Run tests with HTML coverage report"
	@echo "  lint        Ruff linter (fast, replaces flake8)"
	@echo "  format      Ruff auto-formatter (Black-compatible)"
	@echo "  type-check  Mypy static type checker"
	@echo "  dashboard   Streamlit dashboard (port 8501)"
	@echo "  api         FastAPI server (port 8000)"
	@echo "  docker      Build and start both services with Docker Compose"
	@echo "  mlflow      Launch MLflow tracking UI (port 5000)"
	@echo "  clean       Remove generated artefacts (models, cache, .pyc)"
	@echo "  clean-all   clean + remove .venv"
	@echo ""

# ── Environment ───────────────────────────────────────────────────────────────
.PHONY: install
install: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	@echo ">>> Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -r requirements.txt --quiet
	@touch $(VENV)/bin/activate
	@echo ">>> Environment ready. Activate with: source $(VENV)/bin/activate"

# ── Data pipeline ─────────────────────────────────────────────────────────────
.PHONY: data
data: install
	@echo ">>> Running data cleaning notebook..."
	@test -f Resources/bls_state_data.xlsx || \
	  (echo "ERROR: Resources/bls_state_data.xlsx not found." && exit 1)
	@test -f Resources/census_data.csv || \
	  (echo "ERROR: Resources/census_data.csv not found." && exit 1)
	$(JUPYTER) nbconvert --to notebook --execute \
	  --ExecutePreprocessor.timeout=600 \
	  --output high_pay_jobs_data_cleaning.ipynb \
	  high_pay_jobs_data_cleaning.ipynb
	@echo ">>> Cleaned dataset saved to Data/cleaned_high_pay_data.csv"

# ── Model training ────────────────────────────────────────────────────────────
# Uses scripts/train_model.py — no more fragile inline Python one-liners.
.PHONY: model
model: install
	@echo ">>> Training XGBoost salary prediction model..."
	@mkdir -p models
	$(VENV)/bin/python scripts/train_model.py
	@echo ">>> Model artefacts saved to models/"

# ── Testing ───────────────────────────────────────────────────────────────────
.PHONY: test
test: install
	@echo ">>> Running test suite..."
	$(PYTEST) tests/ -v --tb=short

.PHONY: test-fast
test-fast: install
	@echo ">>> Running test suite (quiet)..."
	$(PYTEST) tests/ -q

.PHONY: coverage
coverage: install
	@echo ">>> Running tests with coverage..."
	$(PYTEST) tests/ \
	  --cov=api --cov=pipeline --cov=scripts \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov \
	  -q
	@echo ">>> HTML report: open htmlcov/index.html"

# ── Code quality ──────────────────────────────────────────────────────────────
.PHONY: lint
lint: install
	@echo ">>> Linting with ruff..."
	$(RUFF) check streamlit_app.py pipeline.py api/ tests/ scripts/ \
	  --statistics
	@echo ">>> Lint complete."

.PHONY: format
format: install
	@echo ">>> Formatting with ruff..."
	$(RUFF) format streamlit_app.py pipeline.py api/ tests/ scripts/
	$(RUFF) check streamlit_app.py pipeline.py api/ tests/ scripts/ \
	  --fix --select I  # sort imports
	@echo ">>> Format complete."

.PHONY: type-check
type-check: install
	@echo ">>> Type-checking with mypy..."
	$(MYPY) api/ pipeline.py scripts/ \
	  --ignore-missing-imports \
	  --no-error-summary
	@echo ">>> Type check complete."

# ── Services ──────────────────────────────────────────────────────────────────
.PHONY: dashboard
dashboard: install
	@echo ">>> Starting Streamlit dashboard on http://localhost:8501 ..."
	$(STREAMLIT) run streamlit_app.py \
	  --server.port 8501 \
	  --server.headless true

.PHONY: api
api: install
	@echo ">>> Starting FastAPI server on http://localhost:8000 ..."
	@echo ">>> API docs at http://localhost:8000/docs"
	$(UVICORN) api.main:app --reload --host 0.0.0.0 --port 8000

# ── Docker ────────────────────────────────────────────────────────────────────
.PHONY: docker
docker:
	@echo ">>> Building and starting all services with Docker Compose..."
	docker compose up --build

.PHONY: docker-down
docker-down:
	docker compose down

# ── MLflow UI ────────────────────────────────────────────────────────────────
.PHONY: mlflow
mlflow: install
	@echo ">>> Launching MLflow UI on http://localhost:5000 ..."
	@echo ">>> Run notebook 4 first to populate experiment runs."
	$(VENV)/bin/mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000

# ── Clean ─────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	@echo ">>> Removing generated artefacts..."
	rm -f models/*.ubj models/*.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo ">>> Clean complete."

.PHONY: clean-all
clean-all: clean
	@echo ">>> Removing virtual environment..."
	rm -rf $(VENV)
	@echo ">>> Run 'make install' to recreate the environment."
