"""Ekspor artifact yang dibutuhkan saat inference produksi palm.

Standardisasi fitur hand-crafted (mean dan standar deviasi) serta kosakata
site dihitung sekali dari seluruh manifest palm, lalu disimpan sebagai
fusion_stats.json di folder artifacts. File ini dipakai predict_palm.py agar
tidak perlu membawa manifest dan fitur hand-crafted mentah ke lingkungan
produksi.

Jalankan sekali setiap kali model final berganti:
    python -m app.export_deployment_artifacts_palm
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import artifacts_dir, outputs_dir
from src.common.features import compute_fusion_stats


def main() -> None:
    output_dir = outputs_dir("palm")
    artifact_dir = artifacts_dir("palm")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(output_dir / "manifest.csv")
    manifest = manifest[manifest["roi_precropped"]].reset_index(drop=True)
    handcrafted = pd.read_csv(output_dir / "handcrafted_features.csv")
    site_categories = sorted(manifest["site"].unique().tolist())

    stats = compute_fusion_stats(handcrafted, manifest, site_categories=site_categories)
    payload = {
        "mean": stats["mean"].tolist(),
        "std": stats["std"].tolist(),
        "site_categories": stats["site_categories"],
    }

    output_path = artifact_dir / "fusion_stats.json"
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print("fusion stats saved to", output_path)


if __name__ == "__main__":
    main()
