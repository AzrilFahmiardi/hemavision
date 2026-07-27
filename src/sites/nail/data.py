"""Ingest metadata dataset kuku (Valles-Coral et al. 2025, UNSM Tarapoto, Peru)
dan terjemahkan skema berbahasa Spanyol menjadi Inggris.

Dataset berbagi populasi dan metadata.csv yang sama persis dengan situs palm,
hanya kolom file yang berbeda (UNAS, bukan PALMAS). Berbeda dari palm, folder
unas berisi foto statis JPG satu per partisipan (bukan video), sehingga path
file di-resolve langsung dari ID tanpa fallback stem timestamp seperti palm.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.paths import dataset_root
from src.common.manifest import COLUMNS, adult_threshold, norm_gender, to_float

SITE_NAME = "nail"
DATASET_NAME = "nail_valles_coral"

HB_PLAUSIBLE_RANGE = (3.0, 20.0)
AGE_PLAUSIBLE_MIN = 1.0

SEVERITY_CATEGORIES = ("Normal", "Mild", "Moderate")
SEVERITY_TRANSLATION = {
    "Normal": "Non-Anemic",
    "Mild": "Mild",
    "Moderate": "Moderate",
}

COLUMN_TRANSLATION = {
    "Sexo": "Sex",
    "Edad": "Age",
    "Fatiga": "Fatigue",
    "Debilidad": "Weakness",
    "Latidos irregulares": "Irregular Heartbeat",
    "Dificultad para respirar": "Difficulty Breathing",
    "Mareos o aturdimiento": "Dizziness Or Lightheadedness",
    "Dolor en el pecho": "Chest Pain",
    "Manos y pies fríos": "Cold Hands And Feet",
    "Dolores de cabeza": "Headache",
    "Leve": "Mild",
    "Moderado": "Moderate",
    "Frecuencia Cardiaca": "Heart Rate",
    "Hemoglobina": "Hemoglobin",
    "Oxígeno": "Oxygen",
    "Peso": "Weight",
    "Talla": "Height",
    "YEMAS": "FINGERTIP_FILE",
    "PALMAS": "PALM_FILE",
    "UNAS": "NAIL_FILE",
}

CATEGORY_FOLDER_TRANSLATION = {
    "Leve": "Mild",
    "Moderada": "Moderate",
    "Normal": "Normal",
}


def raw_dir() -> Path:
    """Path folder raw dataset nail (hasil unggah manual dari Google Drive)."""
    return dataset_root(SITE_NAME) / "raw"


def translate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Terjemahkan nama kolom metadata nail dari Spanyol menjadi Inggris."""
    return df.rename(columns=COLUMN_TRANSLATION)


def build_metadata_en(raw_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Baca metadata.csv asli, terjemahkan skemanya, lalu simpan sebagai metadata_en.csv."""
    df = pd.read_csv(raw_csv)
    df_en = translate_metadata(df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_en.to_csv(output_csv, index=False)
    return df_en


def _load_metadata_en() -> pd.DataFrame:
    """Muat metadata_en.csv, membangunnya dari metadata.csv asli bila belum ada."""
    root = raw_dir()
    output_csv = root / "metadata_en.csv"
    if not output_csv.exists():
        return build_metadata_en(root / "metadata.csv", output_csv)
    return pd.read_csv(output_csv)


def _severity_category(record: pd.Series) -> str | None:
    """Tentukan kategori keparahan satu baris dari kolom one-hot Normal/Mild/Moderate.

    Kembalikan None bila tidak ada satupun kolom bernilai 1, atau bila lebih
    dari satu kolom bernilai 1 (baris ambigu, sebaiknya dilewati).
    """
    active = [category for category in SEVERITY_CATEGORIES if to_float(record.get(category)) == 1.0]
    if len(active) != 1:
        return None
    return active[0]


def _resolve_photo_path(root: Path, category: str, identifier: str) -> Path | None:
    """Cari path foto kuku untuk satu baris metadata di folder kategori terkait.

    Berbeda dari palm, file foto konsisten dinamai memakai ID partisipan
    (mis. ID004.jpg) tanpa perlu fallback pencocokan nama file timestamp,
    karena unas diunggah manual dan sudah diverifikasi lengkap 826 file.
    """
    path = root / "unas" / category / f"{identifier}.jpg"
    return path if path.exists() else None


def build_manifest(save: bool = False, output_dir: Path | None = None) -> pd.DataFrame:
    """Bangun manifest dataset nail dari metadata_en.csv dan foto di folder unas.

    Severity diturunkan dari kolom one-hot Normal/Mild/Moderate dan dipetakan ke
    kosakata severity bersama proyek, sedangkan label anemik diturunkan dari
    ambang hemoglobin dewasa sesuai gender, konsisten dengan situs lain. Baris
    tanpa kategori keparahan yang jelas atau tanpa foto yang berhasil diunggah
    dilewati.
    """
    root = raw_dir()
    meta = _load_metadata_en()

    rows = []
    missing = []
    for _, record in meta.iterrows():
        identifier = str(record.get("ID", "")).strip()
        category = _severity_category(record)
        if category is None:
            missing.append(identifier)
            continue
        photo_path = _resolve_photo_path(root, category, identifier)
        if photo_path is None:
            missing.append(identifier)
            continue

        hb = to_float(record.get("Hemoglobin"))
        age = to_float(record.get("Age"))
        if pd.isna(hb) or not (HB_PLAUSIBLE_RANGE[0] <= hb <= HB_PLAUSIBLE_RANGE[1]):
            missing.append(f"{identifier}(hb={hb})")
            continue
        if pd.isna(age) or age < AGE_PLAUSIBLE_MIN:
            missing.append(f"{identifier}(age={age})")
            continue

        gender = norm_gender(record.get("Sex"))
        threshold = adult_threshold(gender)
        rows.append({
            "uid": f"nail_{identifier}",
            "dataset": DATASET_NAME,
            "site": SITE_NAME,
            "roi_path": "",
            "raw_path": str(photo_path),
            "mask_path": "",
            "roi_precropped": False,
            "hb_gdl": hb,
            "age_years": age,
            "gender": gender,
            "severity": SEVERITY_TRANSLATION[category],
            "anemic": int(hb < threshold),
            "hb_threshold": threshold,
        })

    if missing:
        preview = missing[:8]
        print(f"Nail melewati {len(missing)} baris (foto tidak valid atau Hb/Age tidak plausibel): {preview}")

    frame = pd.DataFrame(rows, columns=COLUMNS)
    if save and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_dir / "manifest.csv", index=False)
    return frame


def summarize(frame: pd.DataFrame) -> None:
    """Cetak ringkasan eksplorasi distribusi utama pada manifest nail."""
    print(f"Total sampel: {len(frame)}")
    print("\nDistribusi label anemik:")
    print(frame["anemic"].value_counts().to_string())
    print("\nDistribusi severity:")
    print(frame["severity"].value_counts().to_string())
    print("\nDistribusi gender:")
    print(frame["gender"].value_counts().to_string())
    print("\nStatistik hemoglobin:")
    print(frame["hb_gdl"].agg(["min", "median", "max", "mean"]).round(2).to_string())
    print("\nStatistik umur tahun:")
    print(frame["age_years"].agg(["min", "median", "max"]).round(1).to_string())
