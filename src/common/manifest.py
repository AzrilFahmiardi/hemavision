"""Utilitas manifest generik lintas situs anatomis.

Berisi parsing nilai yang toleran, ambang anemia standar WHO, skema kolom
manifest bersama, dan pembagian data terstratifikasi berbasis pasien. Fungsi
di modul ini tidak menyebut nama dataset atau situs tertentu sehingga dapat
dipakai ulang oleh situs konjungtiva, telapak tangan, atau kuku selama skema
manifestnya konsisten.

Skema kolom manifest:
    uid             identitas unik lintas dataset
    dataset         nama dataset sumber
    site            asal geografis atau sumber pengumpulan data
    roi_path        citra region of interest sebagai RGBA cutout
    raw_path        foto mentah untuk supervisi segmentasi bila tersedia
    mask_path       mask segmentasi bila tersedia
    roi_precropped  True bila ROI sudah tersedia siap pakai
    hb_gdl          kadar hemoglobin dalam g/dL sebagai target regresi bersama
    age_years       umur dalam tahun
    gender          M atau F
    severity        kategori keparahan anemia bila tersedia
    anemic          label biner 0 atau 1 hasil penerapan ambang populasi
    hb_threshold    ambang g/dL yang dipakai untuk label biner sebagai jejak audit
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = [
    "uid", "dataset", "site", "roi_path", "raw_path", "mask_path",
    "roi_precropped", "hb_gdl", "age_years", "gender", "severity",
    "anemic", "hb_threshold",
]


def to_float(value) -> float:
    """Parse angka yang mungkin memakai koma sebagai pemisah desimal."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def norm_gender(value) -> str:
    """Seragamkan penulisan gender menjadi M atau F."""
    if value is None:
        return ""
    text = str(value).strip().upper()
    if text.startswith("M"):
        return "M"
    if text.startswith("F"):
        return "F"
    return ""


def adult_threshold(gender: str) -> float:
    """Ambang anemia dewasa menurut WHO, pria 13 g/dL dan wanita 12 g/dL.

    Ketika gender tidak diketahui, ambang wanita dipakai sebagai default agar
    sensitivitas skrining tetap terjaga.
    """
    return 13.0 if gender == "M" else 12.0


def assign_stratified_split(
    frame: pd.DataFrame,
    fractions=(0.7, 0.15, 0.15),
    seed: int = 42,
) -> pd.DataFrame:
    """Tambahkan kolom split train, val, dan test secara terstratifikasi.

    Stratifikasi dilakukan per kombinasi dataset dan label anemik agar proporsi
    kelas terjaga di setiap partisi. Setiap baris mewakili satu pasien sehingga
    pembagian per baris setara dengan pembagian per pasien.
    """
    rng = np.random.default_rng(seed)
    train_end = fractions[0]
    val_end = fractions[0] + fractions[1]
    result = frame.copy()
    result["split"] = "train"
    for _, group in result.groupby(["dataset", "anemic"]):
        indices = np.array(group.index.to_list())
        rng.shuffle(indices)
        n = len(indices)
        for position, row_index in enumerate(indices):
            ratio = position / n
            if ratio >= val_end:
                result.at[row_index, "split"] = "test"
            elif ratio >= train_end:
                result.at[row_index, "split"] = "val"
    return result


def assign_kfold(frame: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Tambahkan kolom fold berisi indeks 0 hingga n_splits minus 1.

    Stratifikasi dilakukan per kombinasi dataset dan label anemik, mengikuti
    pola assign_stratified_split, agar setiap fold representatif untuk seluruh
    dataset dan kelas yang terlibat. Setiap baris mewakili satu pasien sehingga
    pembagian ini setara dengan pembagian berbasis pasien.
    """
    rng = np.random.default_rng(seed)
    result = frame.copy()
    result["fold"] = 0
    for _, group in result.groupby(["dataset", "anemic"]):
        indices = np.array(group.index.to_list())
        rng.shuffle(indices)
        fold_assignment = np.arange(len(indices)) % n_splits
        for row_index, fold in zip(indices, fold_assignment):
            result.at[row_index, "fold"] = int(fold)
    return result
