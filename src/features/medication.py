"""Medication-related features.

The 23 diabetes drug columns take values: 'No', 'Steady', 'Up', 'Down'.
We derive:
  - per-drug dosage-change flag (Up/Down -> 1)
  - total count of medications the patient is on (Steady/Up/Down)
  - total count of dosage changes across all drugs
"""
import pandas as pd

DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


def add_medication_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    present = [c for c in DRUG_COLS if c in df.columns]

    changed = df[present].isin(["Up", "Down"])
    on_drug = df[present].isin(["Steady", "Up", "Down"])

    df["n_meds_on"] = on_drug.sum(axis=1)
    df["n_med_changes"] = changed.sum(axis=1)
    df["any_med_change"] = (df["n_med_changes"] > 0).astype(int)

    return df
