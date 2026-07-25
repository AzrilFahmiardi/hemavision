"""Pemilihan frame representatif dari video palm.

Video palm merekam partisipan membuka kepalan tangan secara bertahap sehingga
tidak semua frame menampilkan telapak tangan dalam kondisi terbuka dan tajam.
Modul ini memakai MediaPipe HandLandmarker (Tasks API) untuk mendeteksi 21
titik landmark tangan per frame, lalu menilai tiga aspek kualitas yaitu
derajat keterbukaan tangan, ketajaman lewat variance of Laplacian, dan
kecerahan, untuk memilih satu frame terbaik yang dipakai pada tahap
segmentasi ROI berikutnya.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.paths import artifacts_dir

HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

WRIST_INDEX = 0
FINGERTIP_INDICES = (4, 8, 12, 16, 20)

MIN_BRIGHTNESS = 25.0
MAX_BRIGHTNESS = 245.0
OPENNESS_WEIGHT = 0.6
SHARPNESS_WEIGHT = 0.4


def _hand_landmarker_model_path() -> Path:
    """Kembalikan path model HandLandmarker, unduh sekali ke artifacts bila belum ada."""
    path = artifacts_dir("palm") / "hand_landmarker.task"
    if not path.exists():
        urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, path)
    return path


def _hand_openness(landmarks_px: np.ndarray) -> float:
    """Skor derajat keterbukaan tangan.

    Dihitung sebagai jarak rata-rata pergelangan ke lima ujung jari,
    dinormalisasi terhadap diagonal kotak pembatas landmark agar skor tidak
    bergantung pada jarak tangan terhadap kamera.
    """
    wrist = landmarks_px[WRIST_INDEX]
    fingertip_distance = np.linalg.norm(
        landmarks_px[list(FINGERTIP_INDICES)] - wrist, axis=1
    ).mean()
    box_min = landmarks_px.min(axis=0)
    box_max = landmarks_px.max(axis=0)
    diagonal = np.linalg.norm(box_max - box_min) + 1e-6
    return float(fingertip_distance / diagonal)


def _crop_landmarks_box(frame: np.ndarray, landmarks_px: np.ndarray, padding: int = 20) -> np.ndarray:
    """Potong citra ke kotak pembatas landmark tangan dengan padding piksel."""
    height, width = frame.shape[:2]
    x_min = max(0, int(landmarks_px[:, 0].min()) - padding)
    x_max = min(width, int(landmarks_px[:, 0].max()) + padding)
    y_min = max(0, int(landmarks_px[:, 1].min()) - padding)
    y_max = min(height, int(landmarks_px[:, 1].max()) + padding)
    return frame[y_min:y_max, x_min:x_max]


def extract_best_frame(
    video_path: str,
    sample_stride: int = 2,
    min_detection_confidence: float = 0.5,
) -> dict | None:
    """Pilih satu frame terbaik dari video palm untuk segmentasi ROI berikutnya.

    Kandidat dibatasi pada frame dengan tangan terdeteksi MediaPipe
    HandLandmarker. Openness dan sharpness dinormalisasi min-max lintas
    kandidat lalu dijumlahkan berbobot, sedangkan brightness di luar rentang
    wajar memberi penalti pada skor komposit alih-alih membuang kandidat
    sepenuhnya, karena beberapa video punya pencahayaan sedikit di luar ideal
    namun tetap dipakai. Mengembalikan None bila tidak ada satu pun frame
    dengan tangan terdeteksi di seluruh video.
    """
    capture = cv2.VideoCapture(str(video_path))
    candidates = []

    options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_hand_landmarker_model_path())),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=min_detection_confidence,
    )
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        timestamp_ms = 0
        frame_duration_ms = 33
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break
            if frame_index % sample_stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                if result.hand_landmarks:
                    height, width = frame_rgb.shape[:2]
                    landmark = result.hand_landmarks[0]
                    landmarks_px = np.array(
                        [[point.x * width, point.y * height] for point in landmark]
                    )
                    crop = _crop_landmarks_box(frame_rgb, landmarks_px)
                    if crop.size > 0:
                        gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                        candidates.append({
                            "frame_index": frame_index,
                            "frame": frame_rgb,
                            "landmarks_px": landmarks_px,
                            "openness": _hand_openness(landmarks_px),
                            "sharpness": float(cv2.Laplacian(gray_crop, cv2.CV_64F).var()),
                            "brightness": float(gray_crop.mean()),
                        })
            frame_index += 1
            timestamp_ms += frame_duration_ms
    capture.release()

    if not candidates:
        return None

    openness_values = np.array([candidate["openness"] for candidate in candidates])
    sharpness_values = np.array([candidate["sharpness"] for candidate in candidates])
    openness_normalized = (openness_values - openness_values.min()) / (np.ptp(openness_values) + 1e-6)
    sharpness_normalized = (sharpness_values - sharpness_values.min()) / (np.ptp(sharpness_values) + 1e-6)
    in_range = np.array([
        MIN_BRIGHTNESS <= candidate["brightness"] <= MAX_BRIGHTNESS for candidate in candidates
    ])

    composite = OPENNESS_WEIGHT * openness_normalized + SHARPNESS_WEIGHT * sharpness_normalized
    composite = np.where(in_range, composite, composite - 1.0)

    best = candidates[int(np.argmax(composite))]
    return {
        "frame": best["frame"],
        "landmarks_px": best["landmarks_px"],
        "frame_index": best["frame_index"],
        "openness": best["openness"],
        "sharpness": best["sharpness"],
        "brightness": best["brightness"],
    }
