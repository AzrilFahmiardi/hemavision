"""Stage 1. Quality control gate.

Menolak citra berkualitas buruk sebelum masuk pipeline agar terhindar dari efek
garbage in garbage out. Empat metrik dihitung pada area region of interest
konjungtiva, yakni ketajaman melalui variance of Laplacian, kecerahan rata rata,
fraksi pixel glare, dan luas area region of interest.

Loader region of interest bersifat format aware karena dataset memuat beberapa
varian mask, yaitu cutout dengan alpha channel transparan, cutout dengan latar
putih, cutout dengan latar hitam, serta citra 16 bit. Semua varian dikembalikan
sebagai pasangan RGB uint8 dan mask boolean yang seragam.
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@contextlib.contextmanager
def _suppress_c_stderr():
    """Bungkam pesan tingkat C seperti peringatan libpng iCCP selama pembacaan citra.

    Beberapa PNG dataset memiliki profil warna rusak yang memicu peringatan
    berulang dari libpng. Peringatan ditulis langsung ke file descriptor stderr
    sehingga dibungkam dengan mengalihkan descriptor tersebut sementara.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)

DEFAULT_THRESHOLDS = {
    "min_laplacian_variance": 2.0,
    "min_brightness": 25.0,
    "max_brightness": 245.0,
    "max_glare_fraction": 0.30,
    "min_roi_pixels": 300,
}


def load_roi(path: str):
    """Baca citra region of interest dan kembalikan RGB uint8 serta mask boolean.

    Format mask dideteksi otomatis. Bila alpha channel memuat area transparan,
    mask diambil dari alpha. Bila tidak, mask diturunkan dari latar dominan yang
    berupa hitam atau putih. Bila tidak ada latar dominan, seluruh pixel dianggap
    region of interest.
    """
    with _suppress_c_stderr():
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Gagal membaca citra: {path}")
    if image.dtype == np.uint16:
        image = (image / 257.0).round().astype(np.uint8)
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return rgb, np.ones(image.shape[:2], dtype=bool)

    if image.shape[2] == 4:
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = image[:, :, 3]
        if (alpha == 0).mean() > 0.2:
            return rgb, alpha > 10
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    black = rgb.sum(axis=2) <= 15
    white = rgb.min(axis=2) >= 240
    if black.mean() > 0.2:
        return rgb, ~black
    if white.mean() > 0.2:
        return rgb, ~white
    return rgb, np.ones(rgb.shape[:2], dtype=bool)


def erode_mask(mask, erosion: int = 3):
    """Erosi mask boolean untuk menghindari artefak tepi pada perhitungan metrik."""
    if erosion <= 0:
        return mask
    kernel = np.ones((erosion, erosion), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded.astype(bool)


def laplacian_variance(rgb, mask) -> float:
    """Ketajaman citra sebagai variance of Laplacian pada area region of interest."""
    if mask.sum() < 10:
        return 0.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian[mask].var())


def mean_brightness(rgb, mask) -> float:
    """Kecerahan rata rata pada channel V ruang warna HSV di area region of interest."""
    if mask.sum() < 10:
        return 0.0
    value = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 2]
    return float(value[mask].mean())


def glare_fraction(rgb, mask, saturation_level: int = 245) -> float:
    """Fraksi pixel region of interest yang mendekati jenuh sebagai indikasi glare."""
    total = int(mask.sum())
    if total < 10:
        return 1.0
    value = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 2]
    bright = (value >= saturation_level) & mask
    return float(int(bright.sum()) / total)


def quality_check(path: str, thresholds: dict | None = None) -> dict:
    """Evaluasi satu citra dan kembalikan metrik, status lolos, serta alasan."""
    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        limits.update(thresholds)
    rgb, mask = load_roi(path)
    inner = erode_mask(mask, 3)
    metrics = {
        "laplacian_variance": laplacian_variance(rgb, inner),
        "brightness": mean_brightness(rgb, inner),
        "glare_fraction": glare_fraction(rgb, inner),
        "roi_pixels": int(mask.sum()),
    }
    reasons = []
    if metrics["laplacian_variance"] < limits["min_laplacian_variance"]:
        reasons.append("blur")
    if metrics["brightness"] < limits["min_brightness"]:
        reasons.append("underexposed")
    if metrics["brightness"] > limits["max_brightness"]:
        reasons.append("overexposed")
    if metrics["glare_fraction"] > limits["max_glare_fraction"]:
        reasons.append("glare")
    if metrics["roi_pixels"] < limits["min_roi_pixels"]:
        reasons.append("roi_too_small")
    metrics["passed"] = len(reasons) == 0
    metrics["reasons"] = ",".join(reasons)
    return metrics


def run_quality_check(
    frame: pd.DataFrame,
    path_column: str = "roi_path",
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """Jalankan quality check pada seluruh baris manifest dan kembalikan hasilnya."""
    records = []
    for _, row in frame.iterrows():
        result = quality_check(row[path_column], thresholds)
        result["uid"] = row["uid"]
        result["dataset"] = row["dataset"]
        records.append(result)
    columns = ["uid", "dataset", "laplacian_variance", "brightness",
               "glare_fraction", "roi_pixels", "passed", "reasons"]
    return pd.DataFrame(records)[columns]
