"""Clinically-motivated interaction features.

a1c_change_group reproduces the key construct from Strack et al. (2014):
the prognostic value of HbA1c is not the level alone but whether a high
result was acted on by changing diabetes medication.

discharge_group collapses the 30+ discharge_disposition_id codes into
clinically distinct destinations; "discharged home" is the dominant,
lower-risk path while transfers to other facilities carry higher risk.
"""
import pandas as pd

# discharge_disposition_id groupings (death/hospice already removed upstream)
_HOME = {1, 6, 8}                       # home / home health
_TRANSFER = {2, 3, 4, 5, 9, 10, 15,
             16, 17, 22, 23, 24, 27,
             28, 29, 30}                # transferred to another facility


def _discharge_group(code) -> str:
    if pd.isna(code):
        return "Other"
    code = int(code)
    if code in _HOME:
        return "Home"
    if code in _TRANSFER:
        return "Transferred"
    return "Other"


def _a1c_change_group(row) -> str:
    a1c = row.get("A1Cresult", "None")
    changed = row.get("change", "No") == "Ch"
    if a1c in ("None", None) or pd.isna(a1c):
        return "no_test"
    if a1c == "Norm":
        return "normal"
    # high result (>7 or >8)
    return "high_med_changed" if changed else "high_med_unchanged"


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "discharge_disposition_id" in df.columns:
        df["discharge_group"] = df["discharge_disposition_id"].map(
            _discharge_group
        )
    df["a1c_change_group"] = df.apply(_a1c_change_group, axis=1)
    return df
