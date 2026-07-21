"""Stage 0. Ingest dan standardisasi dua dataset (CP-AnemiC dan Eyes-Defy)
menjadi satu manifest terunifikasi.

Modul ini membaca metadata kedua dataset, menyamakan skema kolom, menurunkan
label biner anemia memakai ambang sesuai populasi (anak dan dewasa), serta
menyediakan pembagian data berbasis pasien yang terstratifikasi.

Skema kolom manifest:
    uid             identitas unik lintas dataset
    dataset         cp_anemic atau eyes_defy
    site            ghana, italy, atau india
    roi_path        citra region of interest konjungtiva sebagai RGBA cutout
    raw_path        foto mata utuh untuk supervisi segmentasi, kosong untuk CP-AnemiC
    mask_path       mask palpebral untuk Eyes-Defy, kosong untuk CP-AnemiC
    roi_precropped  True bila ROI sudah tersedia siap pakai
    hb_gdl          kadar hemoglobin dalam g/dL sebagai target regresi bersama
    age_years       umur dalam tahun
    gender          M atau F
    severity        Non-Anemic, Mild, Moderate, atau Severe, hanya untuk CP-AnemiC
    anemic          label biner 0 atau 1 hasil penerapan ambang populasi
    hb_threshold    ambang g/dL yang dipakai untuk label biner sebagai jejak audit
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import cp_anemic_dir, eyes_defy_dir, OUTPUTS

COLUMNS = [
    "uid", "dataset", "site", "roi_path", "raw_path", "mask_path",
    "roi_precropped", "hb_gdl", "age_years", "gender", "severity",
    "anemic", "hb_threshold",
]

PEDIATRIC_THRESHOLD = 11.0


def _to_float(value) -> float:
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


def _norm_gender(value) -> str:
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
            hb = _to_float(record["HB_LEVEL"])
            rows.append({
                "uid": f"cp_{stem}",
                "dataset": "cp_anemic",
                "site": "ghana",
                "roi_path": str(image_path),
                "raw_path": "",
                "mask_path": "",
                "roi_precropped": True,
                "hb_gdl": hb,
                "age_years": round(_to_float(record["Age(Months)"]) / 12.0, 2),
                "gender": _norm_gender(record.get("GENDER")),
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
        meta["Hgb"] = meta["Hgb"].map(_to_float)
        meta["Number"] = meta["Number"].map(_to_float)
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
            gender = _norm_gender(record.get("Gender"))
            hb = _to_float(record["Hgb"])
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
                "age_years": _to_float(record.get("Age")),
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


def build_manifest(save: bool = True) -> pd.DataFrame:
    """Gabungkan kedua dataset menjadi satu manifest dan simpan ke outputs."""
    frame = pd.concat([build_cp_anemic(), build_eyes_defy()], ignore_index=True)
    frame = frame[COLUMNS]
    if save:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        frame.to_csv(OUTPUTS / "manifest.csv", index=False)
    return frame


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


def summarize(frame: pd.DataFrame) -> None:
    """Cetak ringkasan eksplorasi distribusi utama pada manifest."""
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
