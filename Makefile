# ============================================================
# High-Paying Jobs in the US — Reproducibility Makefile
# ============================================================
# Usage:
#   make          — show this help
#   make install  — create venv and install all dependencies
#   make data     — regenerate cleaned dataset from raw sources
#   make model    — train XGBoost model and save to models/
#   make test     — run full pytest suite (57 tests)
#   make lint     — run flake8 on Python source files
#   make dashboard — launch Streamlit on http://localhost:8501
#   make api      — launch FastAPI on http://localhost:8000
#   make docker   — build and start both services with Docker Compose
#   make clean    — remove generated artefacts (models, images, cache)
#   make clean-all — clean + remove the virtual environment
# ============================================================

PYTHON   := python3
VENV     := .venv
PIP      := $(VENV)/bin/pip
PYTEST   := $(VENV)/bin/pytest
FLAKE8   := $(VENV)/bin/flake8
JUPYTER  := $(VENV)/bin/jupyter
STREAMLIT:= $(VENV)/bin/streamlit
UVICORN  := $(VENV)/bin/uvicorn

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
	@echo "  install    Create .venv and install all dependencies"
	@echo "  data       Regenerate cleaned dataset from raw sources"
	@echo "  model      Train XGBoost model and save to models/"
	@echo "  test       Run full pytest suite"
	@echo "  lint       Run flake8 on Python source files"
	@echo "  dashboard  Launch Streamlit dashboard (port 8501)"
	@echo "  api        Launch FastAPI server (port 8000)"
	@echo "  docker     Build and start both services with Docker Compose"
	@echo "  clean      Remove generated artefacts (models, cache, .pyc)"
	@echo "  clean-all  clean + remove .venv"
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
.PHONY: model
model: install
	@echo ">>> Training XGBoost salary prediction model..."
	@mkdir -p models
	$(PYTHON) -c "\
import pickle, yaml, pandas as pd; \
from sklearn.model_selection import train_test_split; \
from xgboost import XGBRegressor; \
cfg = yaml.safe_load(open('config.yaml')); \
edu = cfg['education_order']; \
reg = {s:r for r,ss in cfg['regions'].items() for s in ss}; \
df = pd.read_csv(cfg['data']['cleaned']); \
df['Education_Ord']    = df['Education Level'].map(edu); \
df['Gender_Bin']       = (df['Gender']=='Male').astype(int); \
df['Region']           = df['State Abbreviation'].map(reg); \
df['Region_Code']      = pd.Categorical(df['Region']).codes; \
df['Occ_Mean_Income']  = df.groupby('Occupation')['Annual Income'].transform('mean'); \
df['State_Mean_Income']= df.groupby('State Abbreviation')['Annual Income'].transform('mean'); \
FEAT = ['Age','Education_Ord','Gender_Bin','Region_Code','Employment', \
        'Location Quotient','Jobs per 1000','Hourly Mean','Annual Mean Wage', \
        'Occ_Mean_Income','State_Mean_Income']; \
X,y = df[FEAT], df['Annual Income']; \
Xtr,_,ytr,_ = train_test_split(X,y,test_size=cfg['model']['test_size'],random_state=cfg['model']['random_state']); \
m = XGBRegressor(n_estimators=cfg['model']['n_estimators'],max_depth=cfg['model']['max_depth'], \
    learning_rate=cfg['model']['learning_rate'],subsample=cfg['model']['subsample'], \
    colsample_bytree=cfg['model']['colsample_bytree'],random_state=cfg['model']['random_state'],n_jobs=-1,verbosity=0); \
m.fit(Xtr,ytr); \
pickle.dump(m,open(cfg['model']['model_path'],'wb')); \
pickle.dump(FEAT,open(cfg['model']['features_path'],'wb')); \
print('Model saved to', cfg['model']['model_path'])"
	@echo ">>> Model training complete."

# ── Testing ───────────────────────────────────────────────────────────────────
.PHONY: test
test: install
	@echo ">>> Running test suite..."
	$(PYTEST) tests/ -v --tb=short

.PHONY: test-fast
test-fast: install
	@echo ">>> Running test suite (quiet)..."
	$(PYTEST) tests/ -q

# ── Linting ───────────────────────────────────────────────────────────────────
.PHONY: lint
lint: install
	@echo ">>> Linting with flake8..."
	$(FLAKE8) streamlit_app.py api/ tests/ \
	  --max-line-length=120 \
	  --exclude=__pycache__ \
	  --statistics
	@echo ">>> Lint complete."

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

# ── Clean ─────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	@echo ">>> Removing generated artefacts..."
	rm -f models/*.pkl
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo ">>> Clean complete."

.PHONY: clean-all
clean-all: clean
	@echo ">>> Removing virtual environment..."
	rm -rf $(VENV)
	@echo ">>> Run 'make install' to recreate the environment."
