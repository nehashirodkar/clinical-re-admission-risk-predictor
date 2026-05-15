"""Lightweight EDA script (mirror of notebooks/01_eda.ipynb).

Run after the dataset is downloaded:
    python notebooks/01_eda.py
"""
import pandas as pd

from src.data.load import load_raw

pd.set_option("display.max_columns", 60)


def main() -> None:
    df = load_raw()
    print("Shape:", df.shape)
    print("\n--- Target distribution (raw) ---")
    print(df["readmitted"].value_counts(normalize=True))

    print("\n--- Missingness (top 15) ---")
    miss = df.isna().mean().sort_values(ascending=False)
    print(miss.head(15))

    print("\n--- Encounters per patient ---")
    print(df["patient_nbr"].value_counts().describe())

    print("\n--- Numeric summary ---")
    print(df.select_dtypes("number").describe().T)


if __name__ == "__main__":
    main()
