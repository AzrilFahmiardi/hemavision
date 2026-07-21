"""Resolusi path dataset dan folder output agar proyek berjalan baik di lokal
maupun di server tanpa perubahan kode.

Urutan pencarian folder dataset:
env HEMAVISION_DATASET, lalu dataset di dalam root hemavision (kasus server),
lalu dataset di parent root (kasus authoring lokal di dalam repo proyek).
"""
from __future__ import annotations

import os
from pathlib import Path

HEMAVISION_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = HEMAVISION_ROOT / "outputs"
ARTIFACTS = HEMAVISION_ROOT / "artifacts"


def dataset_root() -> Path:
    """Kembalikan path folder dataset yang valid pertama kali ditemukan."""
    candidates = []
    env = os.environ.get("HEMAVISION_DATASET")
    if env:
        candidates.append(Path(env))
    candidates.append(HEMAVISION_ROOT / "dataset")
    candidates.append(HEMAVISION_ROOT.parent / "dataset")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Folder dataset tidak ditemukan. Set env HEMAVISION_DATASET, atau letakkan "
        f"folder dataset di salah satu lokasi: {[str(c) for c in candidates]}"
    )


def cp_anemic_dir() -> Path:
    """Path folder dataset CP-AnemiC."""
    return dataset_root() / "CP-AnemiC dataset"


def eyes_defy_dir() -> Path:
    """Path folder dataset Eyes-Defy."""
    return dataset_root() / "EYES-DEFY-ANEMIA"
