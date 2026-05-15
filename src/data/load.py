"""Load raw data and apply first-pass cleaning.

Produces an interim cleaned frame:
  - '?' -> NaN
  - drop high-missingness columns
  - drop encounters ending in death / hospice (cannot be readmitted)
  - keep first encounter per patient (independence assumption)
  - build binary 30-day readmission target
"""
import pandas as pd

from src.config import RAW_CSV, INTERIM_DIR, RAW_TARGET, TARGET

# discharge_disposition_id values meaning expired or hospice -> exclude
DEATH_HOSPICE_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

# >40% missing or non-predictive identifiers
DROP_COLS = ["weight", "payer_code", "encounter_id"]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV, na_values="?", low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[~df["discharge_disposition_id"].isin(DEATH_HOSPICE_DISPOSITIONS)]

    # one encounter per patient (first chronologically by encounter_id)
    df = df.sort_values("encounter_id").drop_duplicates(
        subset="patient_nbr", keep="first"
    )

    df[TARGET] = (df[RAW_TARGET] == "<30").astype(int)
    df = df.drop(columns=[RAW_TARGET])

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # gender has a handful of 'Unknown/Invalid' rows
    df = df[df["gender"].isin(["Male", "Female"])]

    return df.reset_index(drop=True)


def main() -> None:
    raw = load_raw()
    print(f"Raw shape: {raw.shape}")
    cleaned = clean(raw)
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Positive rate (30-day readmit): {cleaned[TARGET].mean():.4f}")
    out = INTERIM_DIR / "cleaned.parquet"
    cleaned.to_parquet(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
