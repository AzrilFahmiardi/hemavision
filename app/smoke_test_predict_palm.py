"""Uji asap fungsi predict_palm() dengan video mentah nyata.

Jalankan lokal atau di server (memakai environment hemavision):
    python -m app.smoke_test_predict_palm
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.predict_palm import predict_palm
from src.sites.palm import data


def main() -> None:
    manifest = data.build_manifest(save=False)
    raw_rows = manifest[manifest["raw_path"] != ""].head(5)
    for _, row in raw_rows.iterrows():
        result = predict_palm(
            row["raw_path"], age_years=row["age_years"], gender=row["gender"]
        )
        print(row["uid"], "hb_true", row["hb_gdl"], "->", result)


if __name__ == "__main__":
    main()
