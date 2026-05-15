import pandas as pd

from src.features.medication import add_medication_features
from src.features.diagnoses import _categorize, add_diagnosis_features
from src.features.visits import add_visit_features
from src.features.interactions import add_interaction_features


def test_medication_change_flags():
    df = pd.DataFrame({
        "metformin": ["Steady", "No", "Up"],
        "insulin": ["Up", "No", "Down"],
    })
    out = add_medication_features(df)
    assert out["n_med_changes"].tolist() == [1, 0, 2]
    assert out["n_meds_on"].tolist() == [2, 0, 2]
    assert out["any_med_change"].tolist() == [1, 0, 1]


def test_icd9_categorization():
    assert _categorize("250.83") == "Diabetes"
    assert _categorize("410") == "Circulatory"
    assert _categorize("V45") == "Other"
    assert _categorize(None) == "Missing"
    assert _categorize("486") == "Respiratory"


def test_diagnosis_columns_replaced():
    df = pd.DataFrame({"diag_1": ["250"], "diag_2": ["410"], "diag_3": ["486"]})
    out = add_diagnosis_features(df)
    assert "diag_1" not in out.columns
    assert out["diag_1_cat"].iloc[0] == "Diabetes"


def test_a1c_change_interaction():
    df = pd.DataFrame({
        "A1Cresult": [None, "Norm", ">8", ">7"],
        "change": ["No", "No", "Ch", "No"],
        "discharge_disposition_id": [1, 2, 3, 99],
    })
    out = add_interaction_features(df)
    assert out["a1c_change_group"].tolist() == [
        "no_test", "normal", "high_med_changed", "high_med_unchanged"
    ]
    assert out["discharge_group"].tolist() == [
        "Home", "Transferred", "Transferred", "Other"
    ]


def test_visit_aggregation():
    df = pd.DataFrame({
        "number_outpatient": [1, 0],
        "number_emergency": [2, 0],
        "number_inpatient": [0, 3],
        "age": ["[70-80)", "[40-50)"],
    })
    out = add_visit_features(df)
    assert out["total_prior_visits"].tolist() == [3, 3]
    assert out["had_prior_inpatient"].tolist() == [0, 1]
    assert out["age_mid"].tolist() == [75, 45]
