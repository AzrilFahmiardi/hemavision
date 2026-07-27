"""Segmentasi region of interest fingertip/nail dari foto statis jari.

Berbeda dari palm, MediaPipe HandLandmarker tidak berlaku di sini karena foto
adalah close-up satu jari menyentuh permukaan gelap (diuji langsung pada
sampel dan hanya berhasil mendeteksi tangan pada 1 dari 60 foto), bukan
siluet tangan penuh.

Segmentasi jari sempat memakai threshold Otsu pada grayscale, tapi ini
keliru: ujung jari yang menekan permukaan gelap menimbulkan pantulan cahaya
keputihan pada permukaan itu sendiri (efek yang sama seperti menekan jok
kulit berwarna gelap), dan pantulan itu ikut lolos sebagai foreground karena
sama-sama lebih terang dibanding latar polos. Ditemukan lewat inspeksi visual
langsung pada preview ROI, bukan dari metrik agregat (area jari terlihat
wajar meski maskingnya salah). Diperbaiki dengan thresholding warna kulit
YCbCr (rentang Peksi dkk. 2021, sama seperti src.sites.palm.roi), karena
pantulan tersebut nyaris tidak berwarna (netral/putih) sedangkan kulit
punya rona hangat yang konsisten, sehingga pantulan otomatis tersingkir
tanpa perlu ambang kecerahan terpisah.

Percobaan awal mencari nail plate lewat kontras spekular/desaturasi di
dalam mask jari juga keliru, karena skor tersebut ikut menangkap kilau kulit
di area lain yang tidak terkait kuku sama sekali. Percobaan kedua mencari
kuku sebagai celah warna non-kulit di dalam siluet jari yang solid juga
gagal, karena kuku secara warna tidak cukup berbeda dari kulit sekitarnya,
sehingga celah yang terdeteksi hanya berupa titik kilau kecil, bukan
seluruh area kuku.

Pendekatan akhir murni geometris: protokol akuisisi terverifikasi konsisten
lewat inspeksi manual (ujung jari selalu mengarah ke bagian atas foto,
pangkal jari/tangan keluar dari sisi bawah frame), sehingga area ujung jari
(pita atas siluet jari) dipakai langsung sebagai ROI fingertip/nail tanpa
segmentasi piksel tambahan. Bila lebih dari satu ujung jari masuk pita atas
(jari lain yang tidak fokus di latar belakang ikut tersegmentasi), dipilih
komponen yang posisinya paling ke kanan, karena posisi kolom jari yang
difoto tajam terbukti konsisten lintas sampel (diverifikasi numerik pada
puluhan sampel, variasi hanya sekitar dua persen lebar foto) sedangkan
jari latar belakang bervariasi jauh lebih lebar posisinya.
"""
from __future__ import annotations

import cv2
import numpy as np

YCBCR_LOWER = np.array([0, 77, 133], dtype=np.uint8)
YCBCR_UPPER = np.array([255, 127, 173], dtype=np.uint8)
SKIN_OPEN_KERNEL = 9
SKIN_CLOSE_KERNEL = 25
MIN_FINGER_AREA_FRAC = 0.05
CROP_PADDING_PX = 20
TIP_REGION_FRAC = 0.25


def _skin_color_mask(rgb: np.ndarray) -> np.ndarray:
    """Tandai piksel bernuansa kulit memakai rentang YCbCr baku (Peksi dkk. 2021).

    Rona kulit tetap konsisten meski kecerahannya berubah, sehingga rentang ini
    memisahkan kulit jari dari pantulan cahaya netral/putih pada permukaan
    latar yang justru bisa lebih terang dari kulit itu sendiri.
    """
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    ycbcr = ycrcb[:, :, [0, 2, 1]]
    return cv2.inRange(ycbcr, YCBCR_LOWER, YCBCR_UPPER) > 0


def _fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Isi lubang tertutup di dalam mask lewat flood fill dari sudut citra.

    Kilau spekular pada permukaan kuku kadang cukup terang/desaturasi untuk
    gagal threshold warna kulit meski berada di tengah siluet jari, sehingga
    menyisakan lubang kecil yang terisolasi (tidak menyentuh tepi citra).
    Area latar yang terhubung ke sudul citra tidak ikut terisi, hanya lubang
    yang benar-benar terkurung di dalam siluet.
    """
    height, width = mask_u8.shape
    flood_fill_mask = np.zeros((height + 2, width + 2), np.uint8)
    filled = mask_u8.copy()
    cv2.floodFill(filled, flood_fill_mask, (0, 0), 255)
    holes = cv2.bitwise_not(filled)
    return mask_u8 | holes


def segment_finger(rgb: np.ndarray) -> np.ndarray | None:
    """Segmentasi jari lewat warna kulit YCbCr, bukan kecerahan.

    Opening membuang speckle kecil, closing menyatukan celah tekstur kulit,
    lalu hanya komponen terhubung terbesar yang dipakai sebagai mask jari.
    Lubang tertutup di dalam siluet (misalnya kilau spekular pada permukaan
    kuku yang gagal threshold warna kulit) diisi penuh lewat flood fill,
    karena tujuan mask ini adalah siluet jari, bukan tekstur permukaannya.
    Mengembalikan None bila tidak ditemukan komponen dengan area wajar (foto
    rusak atau tidak ada jari yang terdeteksi).
    """
    skin = _skin_color_mask(rgb)
    open_kernel = np.ones((SKIN_OPEN_KERNEL, SKIN_OPEN_KERNEL), np.uint8)
    opened = cv2.morphologyEx(skin.astype(np.uint8) * 255, cv2.MORPH_OPEN, open_kernel)
    close_kernel = np.ones((SKIN_CLOSE_KERNEL, SKIN_CLOSE_KERNEL), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1:
        return None
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area_frac = stats[largest_label, cv2.CC_STAT_AREA] / skin.size
    if area_frac < MIN_FINGER_AREA_FRAC:
        return None
    finger_mask = (labels == largest_label).astype(np.uint8) * 255
    filled = _fill_holes(finger_mask)
    return filled > 0


def crop_to_mask(rgb: np.ndarray, mask: np.ndarray, padding: int = CROP_PADDING_PX) -> dict:
    """Potong citra dan mask ke bounding box mask dengan padding piksel."""
    height, width = mask.shape
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top = max(0, int(rows[0]) - padding)
    bottom = min(height, int(rows[-1]) + padding + 1)
    left = max(0, int(cols[0]) - padding)
    right = min(width, int(cols[-1]) + padding + 1)
    return {
        "rgb": rgb[top:bottom, left:right],
        "mask": mask[top:bottom, left:right],
    }


def detect_nail_plate(finger_mask: np.ndarray) -> np.ndarray:
    """Ambil area ujung jari (fingertip/nail) dari mask jari secara geometris.

    Nail plate selalu berada di pita atas mask jari sesuai protokol akuisisi
    (lihat catatan modul). Bila lebih dari satu ujung jari masuk pita ini
    (jari lain yang tidak fokus ikut tersegmentasi), komponen paling kanan
    yang dipilih, karena itu konsisten menjadi posisi jari yang difoto tajam.
    """
    rows = np.where(finger_mask.any(axis=1))[0]
    top = int(rows[0])
    bottom = int(rows[-1])
    band_end = top + int(round((bottom - top) * TIP_REGION_FRAC))
    band_mask = np.zeros_like(finger_mask)
    band_mask[top:band_end + 1] = True
    tip_region = finger_mask & band_mask

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        tip_region.astype(np.uint8) * 255, connectivity=8
    )
    if num_labels <= 1:
        return tip_region
    rightmost_label = 1 + int(np.argmax(centroids[1:, 0]))
    return labels == rightmost_label


def polish_saturation_score(rgb: np.ndarray, nail_mask: np.ndarray) -> float:
    """Saturasi rata-rata pada area nail plate, dipakai sebagai sinyal deteksi cat kuku.

    Ambang untuk menandai polish_flag dihitung terpisah di notebook lewat
    metode IQR pada distribusi skor ini di seluruh dataset, bukan ambang tetap
    di modul ini, karena tidak ada label ground truth cat kuku untuk
    dikalibrasi langsung.
    """
    if nail_mask.sum() == 0:
        return 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    return float(saturation[nail_mask].mean())


def build_nail_roi(rgb: np.ndarray) -> dict | None:
    """Bangun ROI fingertip/nail lengkap dari satu foto jari mentah.

    Mengembalikan None bila segmentasi jari gagal total (foto rusak/tidak ada
    jari terdeteksi), ditangani sebagai baris QC gagal di notebook, bukan
    exception yang menghentikan seluruh proses batch.
    """
    finger_mask = segment_finger(rgb)
    if finger_mask is None:
        return None
    nail_mask_full = detect_nail_plate(finger_mask)
    nail_crop = crop_to_mask(rgb, nail_mask_full, padding=CROP_PADDING_PX)
    polish_score = polish_saturation_score(nail_crop["rgb"], nail_crop["mask"])
    return {
        "rgb": nail_crop["rgb"],
        "mask": nail_crop["mask"],
        "finger_area_frac": float(finger_mask.mean()),
        "nail_area_frac": float(nail_mask_full.mean()),
        "polish_score": polish_score,
    }


def save_roi_rgba(path, rgb: np.ndarray, mask: np.ndarray) -> None:
    """Simpan citra ROI sebagai PNG RGBA dengan mask pada alpha channel.

    Format ini konsisten dengan loader src.common.qc.load_roi, sehingga file
    hasil modul ini langsung kompatibel dengan src.common.preprocess tanpa
    penyesuaian tambahan, sama seperti situs palm dan konjungtiva.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    alpha = mask.astype(np.uint8) * 255
    bgra = np.dstack([bgr, alpha])
    cv2.imwrite(str(path), bgra)
