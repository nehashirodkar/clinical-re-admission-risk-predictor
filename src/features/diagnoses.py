"""ICD-9 diagnosis categorization for diag_1, diag_2, diag_3.

Maps raw ICD-9 codes into clinically meaningful groups, following the
grouping used in Strack et al. (2014) for this dataset.
"""
import numpy as np
import pandas as pd

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]


def _categorize(code) -> str:
    if pd.isna(code):
        return "Missing"
    code = str(code)

    # V / E codes -> supplementary
    if code.startswith(("V", "E")):
        return "Other"

    try:
        num = float(code)
    except ValueError:
        return "Other"

    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    if 460 <= num <= 519 or num == 786:
        return "Respiratory"
    if 520 <= num <= 579 or num == 787:
        return "Digestive"
    if int(num) == 250:
        return "Diabetes"
    if 800 <= num <= 999:
        return "Injury"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if 580 <= num <= 629 or num == 788:
        return "Genitourinary"
    if 140 <= num <= 239:
        return "Neoplasms"
    return "Other"


def add_diagnosis_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in DIAG_COLS:
        if col in df.columns:
            df[f"{col}_cat"] = df[col].map(_categorize)
    df = df.drop(columns=[c for c in DIAG_COLS if c in df.columns])
    return df
