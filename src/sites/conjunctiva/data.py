"""Ingest dan standardisasi dua dataset konjungtiva (CP-AnemiC dan Eyes-Defy)
menjadi satu manifest terunifikasi.

Modul ini membaca metadata kedua dataset, menyamakan skema kolom memakai
kontrak generik pada src.common.manifest, menurunkan label biner anemia
memakai ambang sesuai populasi (anak dan dewasa), serta menyiapkan pembagian
data berbasis pasien. Kekhususan situs konjungtiva ada pada cara membaca kedua
dataset ini, misalnya CP-AnemiC berupa strip ter-crop dengan mask pada alpha
channel, sedangkan Eyes-Defy berupa foto mata utuh dengan mask palpebral
terpisah.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.paths import dataset_root
from src.common.manifest import COLUMNS, adult_threshold, norm_gender, to_float

SITE_NAME = "conjunctiva"
PEDIATRIC_THRESHOLD = 11.0


def cp_anemic_dir() -> Path:
    """Path folder dataset CP-AnemiC."""
    return dataset_root(SITE_NAME) / "CP-AnemiC dataset"


def eyes_defy_dir() -> Path:
    """Path folder dataset Eyes-Defy."""
    return dataset_root(SITE_NAME) / "EYES-DEFY-ANEMIA"


def build_cp_anemic() -> pd.DataFrame:
    """Bangun baris manifest untuk dataset CP-AnemiC.

    Citra sudah berupa strip konjungtiva ter-crop dengan mask pada alpha channel,
    sehingga roi_precropped bernilai True dan raw_path dikosongkan.
    """
    root = cp_anemic_dir()
    meta = pd.read_excel(root / "Anemia_Data_Collection_Sheet.xlsx")
    meta["IMAGE_ID"] = meta["IMAGE_ID"].astype(str).str.strip()
    meta = meta.set_index("IMAGE_ID")

    rows = []
    missing = []
    for subdir in ("Anemic", "Non-anemic"):
        for image_path in sorted((root / subdir).glob("*.png")):
            stem = image_path.stem
            if stem not in meta.index:
                missing.append(stem)
                continue
            record = meta.loc[stem]
            hb = to_float(record["HB_LEVEL"])
            rows.append({
                "uid": f"cp_{stem}",
                "dataset": "cp_anemic",
                "site": "ghana",
                "roi_path": str(image_path),
                "raw_path": "",
                "mask_path": "",
                "roi_precropped": True,
                "hb_gdl": hb,
                "age_years": round(to_float(record["Age(Months)"]) / 12.0, 2),
                "gender": norm_gender(record.get("GENDER")),
                "severity": str(record.get("Severity", "")).strip(),
                "anemic": int(hb < PEDIATRIC_THRESHOLD),
                "hb_threshold": PEDIATRIC_THRESHOLD,
            })
    if missing:
        preview = missing[:5]
        print(f"CP-AnemiC melewati {len(missing)} citra tanpa metadata: {preview}")
    return pd.DataFrame(rows)


def _find_eyes_defy_files(folder: Path):
    """Kembalikan pasangan foto mata utuh dan mask palpebral untuk satu folder.

    Palpebral dipilih karena secara literatur merupakan region of interest dengan
    nilai prediktif terbaik.
    """
    raw_candidates = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")))
    palpebral = [
        path for path in folder.glob("*_palpebral.png")
        if "forniceal" not in path.name.lower()
    ]
    raw = raw_candidates[0] if raw_candidates else None
    mask = palpebral[0] if palpebral else None
    return raw, mask


def build_eyes_defy() -> pd.DataFrame:
    """Bangun baris manifest untuk dataset Eyes-Defy.

    Foto mata utuh tersedia sehingga roi_precropped bernilai False, dan mask
    palpebral RGBA dipakai sebagai roi_path yang seragam dengan CP-AnemiC.
    """
    root = eyes_defy_dir()
    rows = []
    for site in ("Italy", "India"):
        site_dir = root / site
        meta = pd.read_excel(site_dir / f"{site}.xlsx")
        meta["Hgb"] = meta["Hgb"].map(to_float)
        meta["Number"] = meta["Number"].map(to_float)
        meta = meta.dropna(subset=["Number", "Hgb"])
        meta["Number"] = meta["Number"].astype(int)
        meta = meta.set_index("Number")

        skipped = []
        folders = [d for d in site_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        for folder in sorted(folders, key=lambda p: int(p.name)):
            number = int(folder.name)
            if number not in meta.index:
                skipped.append(number)
                continue
            raw, mask = _find_eyes_defy_files(folder)
            if raw is None or mask is None:
                skipped.append(number)
                continue
            record = meta.loc[number]
            gender = norm_gender(record.get("Gender"))
            hb = to_float(record["Hgb"])
            threshold = adult_threshold(gender)
            rows.append({
                "uid": f"{site.lower()}_{number:03d}",
                "dataset": "eyes_defy",
                "site": site.lower(),
                "roi_path": str(mask),
                "raw_path": str(raw),
                "mask_path": str(mask),
                "roi_precropped": False,
                "hb_gdl": hb,
                "age_years": to_float(record.get("Age")),
                "gender": gender,
                "severity": "",
                "anemic": int(hb < threshold),
                "hb_threshold": threshold,
            })
        if skipped:
            preview = skipped[:8]
            print(f"Eyes-Defy {site} melewati {len(skipped)} folder tanpa "
                  f"metadata atau file lengkap: {preview}")
    return pd.DataFrame(rows)


def build_manifest(save: bool = False, output_dir: Path | None = None) -> pd.DataFrame:
    """Gabungkan kedua dataset konjungtiva menjadi satu manifest.

    Bila save bernilai True, manifest disimpan ke output_dir. Notebook driver
    sebaiknya mengisi output_dir dengan folder bernamespace situs.
    """
    frame = pd.concat([build_cp_anemic(), build_eyes_defy()], ignore_index=True)
    frame = frame[COLUMNS]
    if save and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_dir / "manifest.csv", index=False)
    return frame


def summarize(frame: pd.DataFrame) -> None:
    """Cetak ringkasan eksplorasi distribusi utama pada manifest konjungtiva."""
    print(f"Total sampel: {len(frame)}")
    print("\nJumlah per dataset dan situs:")
    print(frame.groupby(["dataset", "site"]).size().to_string())
    print("\nDistribusi label anemik per dataset:")
    print(frame.groupby(["dataset", "anemic"]).size().to_string())
    print("\nSeverity pada CP-AnemiC:")
    cp_frame = frame[frame["dataset"] == "cp_anemic"]
    print(cp_frame["severity"].value_counts().to_string())
    print("\nStatistik hemoglobin per dataset:")
    print(frame.groupby("dataset")["hb_gdl"].agg(["min", "median", "max", "mean"]).round(2).to_string())
    print("\nStatistik umur tahun per dataset:")
    print(frame.groupby("dataset")["age_years"].agg(["min", "median", "max"]).round(1).to_string())
