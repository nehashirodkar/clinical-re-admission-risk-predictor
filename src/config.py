"""Central configuration: paths, constants, and project-wide settings."""
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

RAW_CSV = RAW_DIR / "diabetic_data.csv"
IDS_MAPPING_CSV = RAW_DIR / "IDS_mapping.csv"

# --- Target ---
RAW_TARGET = "readmitted"          # original column: '<30', '>30', 'NO'
TARGET = "readmitted_30d"          # engineered binary target: 1 if '<30' else 0

# --- Modeling ---
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
RECALL_TARGET = 0.50               # minimum recall the deployed threshold must achieve

# --- MLflow ---
MLFLOW_EXPERIMENT = "readmission-risk"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
