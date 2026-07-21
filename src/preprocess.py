"""Stage 2. Illumination normalization.

Menstabilkan variasi pencahayaan memakai CLAHE pada channel V ruang warna HSV,
lalu menandai pixel valid dengan membuang pixel sangat gelap atau sangat terang
sebelum ekstraksi fitur. Pendekatan ini mengikuti temuan bahwa normalisasi
pencahayaan mengurangi perbedaan antar perangkat tanpa mengubah pola warna yang
terkait kepucatan.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.qc import load_roi


def crop_to_roi(rgb, mask, padding: int = 4):
    """Potong citra dan mask ke kotak pembatas region of interest dengan padding."""
    if not mask.any():
        return rgb, mask
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top = max(0, rows[0] - padding)
    left = max(0, cols[0] - padding)
    bottom = min(mask.shape[0] - 1, rows[-1] + padding)
    right = min(mask.shape[1] - 1, cols[-1] + padding)
    sub = (slice(top, bottom + 1), slice(left, right + 1))
    return rgb[sub], mask[sub]


def clahe_on_v(rgb, clip_limit: float = 2.0, tile_grid: int = 8):
    """Terapkan CLAHE pada channel V ruang warna HSV lalu kembalikan RGB.

    CLAHE meningkatkan kontras lokal dan meredam pencahayaan tidak merata tanpa
    menggeser informasi kromatisitas pada channel H dan S.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def valid_pixel_mask(rgb, roi, low: int = 20, high: int = 240):
    """Tandai pixel region of interest yang berada dalam rentang intensitas wajar.

    Pixel dengan seluruh channel di bawah low atau di atas high dibuang karena
    kemungkinan berasal dari bayangan, pantulan, atau area bukan jaringan.
    """
    channel_min = rgb.min(axis=2)
    channel_max = rgb.max(axis=2)
    in_range = (channel_min > low) & (channel_max < high)
    return in_range & roi


def normalize_roi(path: str, crop: bool = True, clip_limit: float = 2.0):
    """Pipeline normalisasi untuk satu citra region of interest.

    Mengembalikan RGB ternormalisasi, mask region of interest, dan mask pixel
    valid. Latar di luar region of interest dinolkan agar tidak ikut terhitung
    pada tahap ekstraksi fitur.
    """
    rgb, mask = load_roi(path)
    if crop:
        rgb, mask = crop_to_roi(rgb, mask)
    normalized = clahe_on_v(rgb, clip_limit=clip_limit).copy()
    normalized[~mask] = 0
    valid = valid_pixel_mask(normalized, mask)
    return {
        "rgb": normalized,
        "roi_mask": mask,
        "valid_mask": valid,
    }
