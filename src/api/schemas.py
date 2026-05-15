"""Pydantic request/response models for the prediction API.

The input mirrors the engineered feature frame (post feature-build,
pre-preprocessing). Categorical fields are free-form strings validated
by the model's OneHotEncoder (handle_unknown='ignore').
"""
from pydantic import BaseModel, Field


class RiskDriver(BaseModel):
    feature: str
    shap_value: float
    direction: str


class PatientFeatures(BaseModel):
    race: str | None = None
    gender: str
    age: str = Field(..., examples=["[70-80)"])
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int = Field(..., ge=1, le=14)
    num_lab_procedures: int = Field(..., ge=0)
    num_procedures: int = Field(..., ge=0)
    num_medications: int = Field(..., ge=0)
    number_outpatient: int = Field(..., ge=0)
    number_emergency: int = Field(..., ge=0)
    number_inpatient: int = Field(..., ge=0)
    number_diagnoses: int = Field(..., ge=1)
    max_glu_serum: str | None = None
    A1Cresult: str | None = None
    diag_1_cat: str
    diag_2_cat: str
    diag_3_cat: str
    n_meds_on: int = Field(..., ge=0)
    n_med_changes: int = Field(..., ge=0)
    any_med_change: int = Field(..., ge=0, le=1)
    total_prior_visits: int = Field(..., ge=0)
    had_prior_inpatient: int = Field(..., ge=0, le=1)
    had_prior_emergency: int = Field(..., ge=0, le=1)
    age_mid: int
    change: str
    diabetesMed: str
    medical_specialty: str | None = None


class PredictionResponse(BaseModel):
    risk_score: float = Field(..., description="Calibrated probability [0,1]")
    prediction: int = Field(..., description="1 if >= tuned threshold")
    threshold: float
    top_5_drivers: list[RiskDriver]
