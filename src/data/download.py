"""Download and unpack the UCI Diabetes 130-US Hospitals dataset.

Usage:
    python -m src.data.download
"""
import io
import sys
import zipfile

import requests

from src.config import RAW_DIR, RAW_CSV

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)


def download() -> None:
    if RAW_CSV.exists():
        print(f"Dataset already present at {RAW_CSV} - skipping download.")
        return

    print(f"Downloading dataset from UCI ...")
    resp = requests.get(UCI_ZIP_URL, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as outer:
        outer.extractall(RAW_DIR)
        # UCI ships a nested zip in some mirrors; flatten if needed.
        for name in outer.namelist():
            if name.endswith(".zip"):
                with zipfile.ZipFile(RAW_DIR / name) as inner:
                    inner.extractall(RAW_DIR)

    if not RAW_CSV.exists():
        print(
            "ERROR: diabetic_data.csv not found after extraction. "
            f"Inspect {RAW_DIR} and place files manually.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Done. Files extracted to {RAW_DIR}")


if __name__ == "__main__":
    download()
