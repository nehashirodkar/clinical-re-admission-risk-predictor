"""FastAPI service exposing the readmission risk predictor.

Run:
    uvicorn src.api.main:app --reload
"""
from fastapi import FastAPI

from src.api.predict import predict_patient
from src.api.schemas import PatientFeatures, PredictionResponse

app = FastAPI(
    title="Clinical Re-Admission Risk Predictor",
    description="30-day hospital readmission risk with SHAP explanations.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures) -> PredictionResponse:
    result = predict_patient(patient.model_dump())
    return PredictionResponse(**result)
