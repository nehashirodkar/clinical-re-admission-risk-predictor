"""Orchestrate all feature engineering into a model-ready frame.

Usage:
    python -m src.features.build
"""
import pandas as pd

from src.config import INTERIM_DIR, PROCESSED_DIR, TARGET
from src.features.medication import add_medication_features, DRUG_COLS
from src.features.diagnoses import add_diagnosis_features
from src.features.visits import add_visit_features
from src.features.interactions import add_interaction_features

# identifiers / leakage-prone columns dropped before modeling
DROP_AFTER = ["patient_nbr"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_medication_features(df)
    df = add_diagnosis_features(df)
    df = add_visit_features(df)
    df = add_interaction_features(df)

    # raw per-drug columns are now summarized; drop to reduce dimensionality
    df = df.drop(columns=[c for c in DRUG_COLS if c in df.columns])
    df = df.drop(columns=[c for c in DROP_AFTER if c in df.columns])
    return df


def main() -> None:
    src = INTERIM_DIR / "cleaned.parquet"
    df = pd.read_parquet(src)
    print(f"Loaded {df.shape} from {src}")

    feats = build_features(df)
    print(f"Feature frame: {feats.shape}")
    print(f"Columns: {sorted(feats.columns)}")

    out = PROCESSED_DIR / "features.parquet"
    feats.to_parquet(out, index=False)
    print(f"Wrote {out} (target={TARGET}, pos_rate={feats[TARGET].mean():.4f})")


if __name__ == "__main__":
    main()
