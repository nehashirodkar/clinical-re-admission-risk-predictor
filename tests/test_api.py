"""API smoke tests. Skipped automatically if the model hasn't been trained."""
import pytest
from fastapi.testclient import TestClient

from src.config import MODELS_DIR

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "pipeline.joblib").exists(),
    reason="model not trained yet (run python -m src.models.train)",
)

SAMPLE = {
    "race": "Caucasian", "gender": "Female", "age": "[70-80)",
    "admission_type_id": 1, "discharge_disposition_id": 1,
    "admission_source_id": 7, "time_in_hospital": 5,
    "num_lab_procedures": 41, "num_procedures": 1, "num_medications": 18,
    "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 1,
    "number_diagnoses": 9, "max_glu_serum": "None", "A1Cresult": "None",
    "diag_1_cat": "Circulatory", "diag_2_cat": "Diabetes",
    "diag_3_cat": "Other", "n_meds_on": 2, "n_med_changes": 1,
    "any_med_change": 1, "total_prior_visits": 1,
    "had_prior_inpatient": 1, "had_prior_emergency": 0, "age_mid": 75,
    "change": "Ch", "diabetesMed": "Yes", "medical_specialty": "Missing",
}


@pytest.fixture(scope="module")
def client():
    from src.api.main import app

    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_shape(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert len(body["top_5_drivers"]) == 5


def test_predict_validation_error(client):
    bad = {**SAMPLE}
    del bad["gender"]
    assert client.post("/predict", json=bad).status_code == 422
