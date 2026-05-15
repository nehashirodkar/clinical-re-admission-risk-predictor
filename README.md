# Clinical Re-Admission Risk Predictor

Predicts 30-day hospital readmission risk on the UCI Diabetes 130-US Hospitals
dataset (~100K encounters) using XGBoost with SMOTE oversampling, threshold
optimization for clinical recall, MLflow experiment tracking, and SHAP-based
per-patient explainability served via FastAPI + a Streamlit dashboard.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Quickstart

From the project root, with the venv activated:

```bash
python -m src.data.download      # fetch UCI dataset into data/raw/
python -m src.data.load          # clean -> data/interim/cleaned.parquet
python -m src.features.build     # feature-engineer -> data/processed/features.parquet
python -m src.models.train       # XGBoost+SMOTE, 5-fold CV, MLflow, threshold; saves models/
pytest -q                        # run the test suite

uvicorn src.api.main:app --reload          # API on http://localhost:8000
streamlit run dashboard/app.py             # dashboard (separate terminal)
mlflow ui --backend-store-uri ./mlruns     # experiment tracking UI
```

If `python -m src.<...>` raises `ModuleNotFoundError: No module named 'src'`,
prefix the command with the repo root on the path:
`PYTHONPATH=. python -m src.models.train` (or `set PYTHONPATH=.` on Windows cmd).

## Dataset

`python -m src.data.download` fetches "Diabetes 130-US Hospitals for Years
1999-2008" from the UCI ML Repository automatically into `data/raw/`
(`diabetic_data.csv`, `IDS_mapping.csv`). The `data/` directory is gitignored
and recreated on first run.

## Results

CV AUC ≈ 0.64, test AUC ≈ 0.64 — consistent with the published ceiling for
this dataset (Strack et al., 2014); materially higher numbers on this data
typically indicate leakage. Engineering the paper's HbA1c × medication-change
interaction did not raise AUC, which corroborates that ceiling.

The decision threshold implements an explicit clinical rule: **maximize
precision subject to recall ≥ 0.50**, reflecting the cost asymmetry of
missing a readmission. At that operating point the held-out test set gives
recall ≈ 0.50, precision ≈ 0.14. Note this recall is an operating-point
choice on a fixed model, distinct from the model's intrinsic ranking
ability (AUC).

## Project layout

```
src/
  config.py        paths & constants
  data/            load + preprocess
  features/        medication / diagnoses / visits feature engineering
  models/          train, evaluate, threshold tuning (XGBoost + SMOTE + MLflow)
  explain/         SHAP TreeExplainer utilities
  api/             FastAPI service (Pydantic-validated)
dashboard/         Streamlit app
notebooks/         EDA & experimentation
tests/             pytest suite
```

## Pipeline

0. Setup & dataset acquisition
1. EDA & data cleaning (first-encounter dedup, drop death/hospice discharges)
2. Feature engineering (medication change flags, ICD-9 grouping, visit aggregates)
3. Modeling (XGBoost + SMOTE, 5-fold stratified CV, MLflow, F2 threshold tuning)
4. SHAP explainability (global importances + per-patient top-5 drivers)
5. FastAPI + Streamlit serving
6. Tests & docs
