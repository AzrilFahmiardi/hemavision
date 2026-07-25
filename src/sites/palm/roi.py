"""Segmentasi region of interest palm dari landmark MediaPipe Hands.

Tidak ada mask ground truth untuk telapak tangan pada dataset ini sehingga
segmentasi tidak melatih model baru seperti pada situs konjungtiva. Sebagai
gantinya, region of interest dibangun dari convex hull titik landmark tangan
yang mencakup telapak, pangkal jari, dan pergelangan, lalu diperhalus dengan
thresholding warna kulit YCbCr mengikuti pola Peksi dkk. (2021) untuk membuang
noise seperti bayangan atau gelang yang ikut masuk convex hull.
"""
from __future__ import annotations

import cv2
import numpy as np

YCBCR_LOWER = np.array([0, 133, 77], dtype=np.uint8)
YCBCR_UPPER = np.array([255, 173, 127], dtype=np.uint8)
HULL_DILATION_PX = 8
MIN_SKIN_PIXELS = 300


def _convex_hull_mask(frame_shape: tuple, landmarks_px: np.ndarray) -> np.ndarray:
    """Bangun mask convex hull dari landmark tangan, didilasi agar mencakup tepi telapak."""
    height, width = frame_shape[:2]
    hull = cv2.convexHull(landmarks_px.astype(np.int32))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if HULL_DILATION_PX > 0:
        kernel = np.ones((HULL_DILATION_PX, HULL_DILATION_PX), np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask > 0


def _skin_color_mask(rgb: np.ndarray) -> np.ndarray:
    """Tandai pixel bernuansa kulit memakai rentang YCbCr baku (Peksi dkk. 2021)."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    ycbcr = ycrcb[:, :, [0, 2, 1]]
    return cv2.inRange(ycbcr, YCBCR_LOWER, YCBCR_UPPER) > 0


def palm_mask_from_landmarks(frame: np.ndarray, landmarks_px: np.ndarray) -> dict:
    """Bangun mask region of interest palm dan kembalikan citra ter-crop beserta mask.

    Mask akhir adalah irisan convex hull landmark dengan mask warna kulit
    YCbCr, sehingga area di luar hull maupun area dalam hull yang bukan kulit
    tidak ikut terhitung pada ekstraksi fitur berikutnya. Bila irisan terlalu
    kecil (misalnya karena pencahayaan membuat threshold warna kulit gagal),
    mask jatuh kembali ke convex hull murni agar tetap ada region yang dipakai.
    """
    hull_mask = _convex_hull_mask(frame.shape, landmarks_px)
    skin_mask = _skin_color_mask(frame)
    combined_mask = hull_mask & skin_mask
    if combined_mask.sum() < MIN_SKIN_PIXELS:
        combined_mask = hull_mask

    rows = np.where(combined_mask.any(axis=1))[0]
    cols = np.where(combined_mask.any(axis=0))[0]
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]
    cropped_rgb = frame[top:bottom + 1, left:right + 1]
    cropped_mask = combined_mask[top:bottom + 1, left:right + 1]
    return {"rgb": cropped_rgb, "mask": cropped_mask}


def save_roi_rgba(path, rgb: np.ndarray, mask: np.ndarray) -> None:
    """Simpan citra ROI sebagai PNG RGBA dengan mask pada alpha channel.

    Format ini konsisten dengan loader src.common.qc.load_roi yang mendeteksi
    mask dari alpha channel, sehingga file hasil modul ini langsung kompatibel
    dengan src.common.preprocess.normalize_roi tanpa penyesuaian tambahan.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    alpha = mask.astype(np.uint8) * 255
    bgra = np.dstack([bgr, alpha])
    cv2.imwrite(str(path), bgra)
