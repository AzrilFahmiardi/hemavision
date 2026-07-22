"""Resolusi path dataset dan folder output agar proyek berjalan baik di lokal
maupun di server tanpa perubahan kode.

Dataset, output, dan artifact dinamespace per situs anatomis (misalnya
conjunctiva, palm, nail) agar pengembangan satu situs tidak menimpa atau
bergantung pada situs lain, sesuai arsitektur multi-situs proyek ini.

Urutan pencarian folder dataset dasar:
env HEMAVISION_DATASET, lalu dataset di dalam root hemavision (kasus server),
lalu dataset di parent root (kasus authoring lokal di dalam repo proyek).
"""
from __future__ import annotations

import os
from pathlib import Path

HEMAVISION_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = HEMAVISION_ROOT / "outputs"
ARTIFACTS = HEMAVISION_ROOT / "artifacts"


def dataset_root(site: str) -> Path:
    """Kembalikan path folder dataset milik satu situs anatomis tertentu."""
    candidates = []
    env = os.environ.get("HEMAVISION_DATASET")
    if env:
        candidates.append(Path(env) / site)
    candidates.append(HEMAVISION_ROOT / "dataset" / site)
    candidates.append(HEMAVISION_ROOT.parent / "dataset" / site)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Folder dataset situs '{site}' tidak ditemukan. Set env HEMAVISION_DATASET, "
        f"atau letakkan folder dataset di salah satu lokasi: {[str(c) for c in candidates]}"
    )


def outputs_dir(site: str) -> Path:
    """Kembalikan folder output milik satu situs, dibuat otomatis bila belum ada."""
    path = OUTPUTS / site
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifacts_dir(site: str) -> Path:
    """Kembalikan folder artifact milik satu situs, dibuat otomatis bila belum ada."""
    path = ARTIFACTS / site
    path.mkdir(parents=True, exist_ok=True)
    return path
