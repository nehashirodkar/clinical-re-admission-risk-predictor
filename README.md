---
title: Clinical Readmission Risk Predictor
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# Clinical Re-Admission Risk Predictor

**A 30-day hospital readmission risk model on the UCI Diabetes 130-US Hospitals dataset — XGBoost + SMOTE, recall-targeted thresholding, per-patient SHAP explanations, served via a Pydantic-validated FastAPI + a clinical Streamlit dashboard, deployed on Hugging Face.**

<br/>

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-gradient%20boosting-017CEE)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-F7931E?logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-explainability-1F77B4)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Hugging%20Face-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-8%20passing-2ea44f)

[Architecture](#-architecture) · [Demo](#-demo) · [Quickstart](#-quickstart) · [Results](#-measured-results) · [Pipeline](#-pipeline)

</div>

---

## ⟡ What it is

**Clinical Re-Admission Risk Predictor** estimates the probability that a diabetic inpatient is re-admitted within 30 days of discharge, and surfaces the factors driving each individual prediction. An **XGBoost** classifier trained with **SMOTE** oversampling and **5-fold stratified CV** scores each encounter; the decision threshold is tuned to an explicit clinical rule (maximise precision subject to recall ≥ 0.50); a **SHAP TreeExplainer** returns the top-5 risk drivers per patient. The model is served behind a **Pydantic-validated FastAPI** endpoint and a clinical **Streamlit** dashboard.

> **Status: complete + deployed.** End-to-end pipeline (download → clean → features → train → explain), MLflow experiment tracking, 8 automated tests, and a live containerised demo on Hugging Face Spaces — validated end-to-end in production.

## ⟡ Architecture

```
   patient encounter (Streamlit form)
                │
                ▼
   ┌────────────────────────┐
   │  FastAPI /predict       │  Pydantic-validated request
   └───────────┬─────────────┘
               │
   ┌───────────▼─────────────┐
   │  feature regeneration    │  interaction features rebuilt server-side
   └───────────┬─────────────┘
               │
   ┌───────────▼─────────────┐
   │  sklearn ColumnTransformer│  impute · scale · one-hot
   └───────────┬─────────────┘
               │
   ┌───────────▼─────────────┐      ┌─────────────────────┐
   │  XGBoost classifier      │────► │ SHAP TreeExplainer   │
   │  (SMOTE-trained)         │      │ top-5 risk drivers   │
   └───────────┬─────────────┘      └──────────┬──────────┘
               │                                │
               ▼                                ▼
   ┌──────────────────────────────────────────────────────┐
   │  response: risk score · decision · top-5 drivers       │
   │  → Streamlit gauge + risk band + factor cards          │
   └──────────────────────────────────────────────────────┘
```

## ⟡ Demo

**Live:** https://huggingface.co/spaces/NehaS98/clinical-readmission-risk-predictor

Enter a patient encounter (demographics, current stay, prior utilisation, ICD-9 diagnosis groups) and the dashboard returns the estimated 30-day readmission probability on a risk gauge, a colour-coded decision band, and the top contributing factors for that patient.

| Signal | Typical effect on risk |
|---|---|
| High prior **inpatient** visits | strongest single risk driver in the model |
| Discharge disposition / destination | materially shifts risk |
| HbA1c result × medication change | the Strack et al. (2014) interaction, engineered explicitly |
| Number of diagnoses, time in hospital | secondary contributors |

## ⟡ Quickstart

```bash
# 1. Install
python -m venv .venv
.venv\Scripts\activate                 # Windows
pip install -r requirements.txt

# 2. Build data + model
python -m src.data.download            # fetch UCI dataset into data/raw/
python -m src.data.load                # clean  -> data/interim/cleaned.parquet
python -m src.features.build           # features -> data/processed/features.parquet
python -m src.models.train             # XGBoost+SMOTE, 5-fold CV, MLflow, threshold

# 3. Test + serve
pytest -q
uvicorn src.api.main:app --reload      # API on http://localhost:8000
streamlit run dashboard/app.py         # dashboard (separate terminal)
mlflow ui --backend-store-uri ./mlruns # experiment tracking UI
```

> If `python -m src.<...>` raises `ModuleNotFoundError: No module named 'src'`, prefix with the repo root: `PYTHONPATH=. python -m src.models.train` (or `set PYTHONPATH=.` on Windows cmd).

The UCI "Diabetes 130-US Hospitals for Years 1999-2008" dataset (~100K encounters) is fetched automatically into `data/raw/`; the `data/` directory is gitignored and recreated on first run.

## ⟡ Measured results

- **Test AUC ≈ 0.64 · CV AUC ≈ 0.64 (±0.005)** — consistent with the published ceiling for this dataset (Strack et al., 2014). Materially higher numbers here typically indicate leakage; engineering the paper's HbA1c × medication-change interaction did **not** raise AUC, which corroborates the ceiling.
- **Recall ≈ 0.50 · precision ≈ 0.14** at the deployed threshold — an explicit operating-point choice (maximise precision subject to recall ≥ 0.50), reflecting the cost asymmetry of missing a readmission. This recall is a threshold policy on a fixed model, distinct from the model's intrinsic ranking ability (AUC).
- **8/8 tests passing** — feature engineering, ICD-9 categorisation, the Strack interaction, and the FastAPI prediction contract.
- **Honest evaluation** — first-encounter-only dedup (the statistically correct choice; lowers headline positive rate ~11% → ~9%); no target leakage.

| Stage | Metric |
|---|---:|
| Cross-validation (5-fold) | AUC 0.64 ± 0.005 |
| Held-out test | AUC 0.64 |
| Test @ deployed threshold | recall 0.50 · precision 0.14 |

## ⟡ Pipeline

| Stage | What happens |
|---|---|
| **Acquisition** | Auto-download UCI Diabetes 130-US Hospitals dataset |
| **Cleaning** | First-encounter dedup; drop death/hospice discharges; build binary 30-day target |
| **Features** | Medication-change flags, ICD-9 diagnosis grouping, visit aggregates, HbA1c × med-change + discharge-destination interactions |
| **Modeling** | XGBoost + SMOTE, 5-fold stratified CV, MLflow tracking, recall-targeted threshold |
| **Explainability** | SHAP TreeExplainer — global importances + per-patient top-5 drivers |
| **Serving** | Pydantic-validated FastAPI `/predict` + clinical Streamlit dashboard |

## ⟡ Project structure

```
src/
├── config.py        # paths & constants
├── data/            # download, load/clean, preprocessing pipeline
├── features/        # medication · diagnoses · visits · interactions
├── models/          # train (XGBoost+SMOTE+MLflow), evaluate, threshold
├── explain/         # SHAP TreeExplainer utilities
└── api/             # FastAPI service + Pydantic schemas
dashboard/           # clinical Streamlit app (risk gauge + drivers)
notebooks/           # EDA (executed, with plots)
tests/               # 8 PyTest cases (features + API)
Dockerfile           # Hugging Face Spaces (Docker SDK)
```

## ⟡ Deployment

| Target | How |
|---|---|
| **Local** | `uvicorn src.api.main:app` + `streamlit run dashboard/app.py` |
| **Docker** | Builds FastAPI (internal :8000) + Streamlit (:7860) in one image |
| **Hugging Face Spaces** | Push to the `hf` remote → auto-rebuild; model shipped via Git LFS |

## ⟡ Disclaimer

Research / portfolio demonstration only. Trained on a public dataset with a known performance ceiling; **not a medical device and not for clinical use.**
