"""Visit-history aggregation and misc clinical features."""
import pandas as pd

# age is given as bracket strings like '[70-80)'; use bracket midpoint
_AGE_MIDPOINTS = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}


def add_visit_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["had_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)
    df["had_prior_emergency"] = (df["number_emergency"] > 0).astype(int)

    if "age" in df.columns:
        df["age_mid"] = df["age"].map(_AGE_MIDPOINTS)

    # lab/result categoricals: collapse to ordered-ish buckets
    for col in ("max_glu_serum", "A1Cresult"):
        if col in df.columns:
            df[col] = df[col].fillna("None")

    return df
