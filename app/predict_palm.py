"""Modul inference untuk model palm.

Membungkus pipeline lengkap situs palm, yaitu pemilihan frame representatif
dari video mentah, segmentasi region of interest berbasis landmark tangan dan
warna kulit, quality control, normalisasi iluminasi, ekstraksi fitur
dual-path, dan prediksi multi-task, menjadi satu fungsi predict_palm(). Fungsi
ini terpisah dari app/predict.py (konjungtiva) karena input mentahnya video
dan tiga stage awal pipeline berbeda total, sedangkan Stage 3 ke atas dipakai
bersama lewat src/common/. Seluruh model memakai checkpoint deployment-ready
yang sudah tersimpan di artifacts/palm, tidak melatih ulang apa pun. Sumber
daya (model dan statistik) dimuat sekali secara malas pada pemanggilan
pertama, lalu dipakai ulang untuk pemanggilan berikutnya agar inference
berikutnya cepat.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import cv2
import json
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import artifacts_dir, outputs_dir
from src.common import features, preprocess, qc
from src.common.models import HemavisionModel, SEVERITY_ORDER
from src.sites.palm import frame_selection, roi as palm_roi

_ARTIFACT_DIR = artifacts_dir("palm")
_OUTPUT_DIR = outputs_dir("palm")

_resources: dict = {}


def _load_resources(device: str = "cpu") -> dict:
    """Muat model dan statistik sekali, simpan di cache modul untuk dipakai ulang."""
    if _resources:
        return _resources

    embedding_backbone = features.EmbeddingBackbone(backbone_name="resnet18")
    embedding_backbone.load_state_dict(
        torch.load(_ARTIFACT_DIR / "embedding_backbone_resnet18.pt", map_location=device)
    )
    embedding_backbone.eval()

    with open(_ARTIFACT_DIR / "fusion_stats.json") as handle:
        stats_raw = json.load(handle)
    fusion_stats = {
        "mean": np.array(stats_raw["mean"], dtype=np.float32).reshape(-1),
        "std": np.array(stats_raw["std"], dtype=np.float32).reshape(-1),
        "site_categories": stats_raw["site_categories"],
    }

    input_dim = len(features.HANDCRAFTED_COLUMNS) + features.EMBEDDING_DIM + 2 + len(fusion_stats["site_categories"])
    prediction_model = HemavisionModel(input_dim=input_dim)
    prediction_model.load_state_dict(
        torch.load(_ARTIFACT_DIR / "multitask_full_fusion_resnet18_fold0.pt", map_location=device)
    )
    prediction_model.eval()

    _resources.update({
        "embedding_backbone": embedding_backbone,
        "prediction_model": prediction_model,
        "fusion_stats": fusion_stats,
    })
    return _resources


def _check_quality(rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Hitung metrik region of interest dan alasan penolakan bila ada.

    Berbeda dari konjungtiva, gate blur/brightness/glare (qc.DEFAULT_THRESHOLDS)
    tidak dipakai di sini. Threshold tersebut dikalibrasi untuk foto mata dan
    tidak berlaku untuk kulit telapak tangan, yang secara alami memantulkan
    cahaya jauh lebih terang. Manifest pelatihan palm sendiri hanya menandai
    roi_precropped berdasarkan berhasil-tidaknya ekstraksi ROI
    (notebooks/palm/03_roi_segmentation.ipynb), bukan lolos-tidaknya gate
    kualitas semacam ini, sehingga model produksi memang dilatih pada seluruh
    ROI yang berhasil diekstrak tanpa filter blur/brightness/glare. Satu-satunya
    kegagalan yang relevan di sini adalah ROI yang terlalu kecil untuk diproses.
    """
    inner = qc.erode_mask(mask, 3)
    metrics = {
        "laplacian_variance": qc.laplacian_variance(rgb, inner),
        "brightness": qc.mean_brightness(rgb, inner),
        "glare_fraction": qc.glare_fraction(rgb, inner),
        "roi_pixels": int(mask.sum()),
    }
    reasons = []
    if metrics["roi_pixels"] < qc.DEFAULT_THRESHOLDS["min_roi_pixels"]:
        reasons.append("roi_too_small")
    metrics["passed"] = len(reasons) == 0
    metrics["reasons"] = reasons
    return metrics


def _encode_png_base64(rgb: np.ndarray) -> str:
    """Encode citra RGB sebagai PNG base64, dipakai untuk menyisipkan tahap pipeline ke response API."""
    success, buffer = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("ascii")


def _draw_landmarks_overlay(rgb: np.ndarray, landmarks_px: np.ndarray | None) -> np.ndarray:
    """Gambar 21 titik landmark MediaPipe Hands di atas frame mentah, untuk keperluan visual.

    Bila frame terpilih lewat fallback skin-color (tanpa landmark valid), frame
    dikembalikan apa adanya tanpa overlay.
    """
    overlay = rgb.copy()
    if landmarks_px is None:
        return overlay
    for point in landmarks_px:
        cv2.circle(overlay, (int(point[0]), int(point[1])), 4, (255, 0, 0), -1)
    return overlay


def _draw_mask_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Gambar kontur region of interest hasil segmentasi di atas citra asli, untuk keperluan visual."""
    overlay = rgb.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)
    return overlay


def _biomarker_heatmap(rgb: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Hasilkan heatmap erythema index (log rasio kanal hijau), mengikuti formula notebook 07."""
    green = rgb[:, :, 1].astype(np.float32)
    erythema_map = np.log10(1.0 / (green / 255.0 + 1e-6))
    valid_values = erythema_map[valid_mask]
    low, high = valid_values.min(), valid_values.max()
    normalized = np.clip((erythema_map - low) / (high - low + 1e-6), 0, 1)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid_mask] = 0
    return colored


def predict_palm(
    video_path: str,
    age_years: float,
    gender: str,
    device: str = "cpu",
    include_stages: bool = False,
) -> dict:
    """Prediksi hemoglobin, status anemia, dan severity dari video telapak tangan mentah.

    Bila tidak ada satu pun frame dengan tangan terdeteksi (baik lewat
    landmark maupun fallback skin-color), hasil berisi frame_detected bernilai
    False dan prediksi tidak dihitung. Bila frame ditolak quality control,
    hasil berisi passed_qc bernilai False beserta alasan penolakan. Bila
    lolos, hasil berisi estimasi hemoglobin dalam g/dL, status anemia dengan
    probabilitas, dan severity dengan catatan keandalan yang masih rendah.

    Bila include_stages bernilai True, hasil turut menyertakan key stage_images
    berisi citra tiap tahap pipeline (raw_frame, landmarks, roi_segmented,
    illumination_normalized, biomarker_heatmap) sebagai PNG ter-encode base64.
    """
    resources = _load_resources(device)

    best_frame = frame_selection.extract_best_frame(video_path)
    if best_frame is None:
        return {"frame_detected": False, "reasons": ["no_hand_detected"]}

    raw_rgb = best_frame["frame"]
    landmarks_px = best_frame["landmarks_px"]
    if landmarks_px is not None:
        segmented = palm_roi.palm_mask_from_landmarks(raw_rgb, landmarks_px)
    else:
        segmented = palm_roi.palm_mask_skin_only(raw_rgb)
    rgb, mask = segmented["rgb"], segmented["mask"]
    rgb, mask = preprocess.crop_to_roi(rgb, mask)

    quality = _check_quality(rgb, mask)
    if not quality["passed"]:
        result = {
            "frame_detected": True,
            "detection_method": best_frame["detection_method"],
            "passed_qc": False,
            "reasons": quality["reasons"],
            "metrics": quality,
        }
        if include_stages:
            result["stage_images"] = {
                "raw_frame": _encode_png_base64(raw_rgb),
                "landmarks": _encode_png_base64(_draw_landmarks_overlay(raw_rgb, landmarks_px)),
                "roi_segmented": _encode_png_base64(_draw_mask_overlay(rgb, mask)),
            }
        return result

    normalized_rgb = preprocess.clahe_on_v(rgb).copy()
    normalized_rgb[~mask] = 0
    valid_mask = preprocess.valid_pixel_mask(normalized_rgb, mask)

    handcrafted_values = features.compute_handcrafted_features(normalized_rgb, valid_mask)
    handcrafted_vector = np.array(
        [handcrafted_values[column] for column in features.HANDCRAFTED_COLUMNS], dtype=np.float32
    )
    stats = resources["fusion_stats"]
    handcrafted_standardized = (handcrafted_vector - stats["mean"]) / stats["std"]

    resized = cv2.resize(normalized_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    resized_float = resized.astype(np.float32) / 255.0
    resized_float = (resized_float - features.IMAGENET_MEAN) / features.IMAGENET_STD
    image_tensor = torch.from_numpy(resized_float.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resources["embedding_backbone"](image_tensor).cpu().numpy()[0]

    age_normalized = np.array([age_years / 100.0], dtype=np.float32)
    gender_binary = np.array([1.0 if gender.upper().startswith("M") else 0.0], dtype=np.float32)
    site_dummies = np.array(
        [1.0 if category == "palm" else 0.0 for category in stats["site_categories"]], dtype=np.float32
    )

    fusion_vector = np.concatenate(
        [handcrafted_standardized, embedding, age_normalized, gender_binary, site_dummies]
    )[None, :]

    with torch.no_grad():
        outputs = resources["prediction_model"](torch.tensor(fusion_vector, dtype=torch.float32, device=device))
        predicted_hb = float(outputs["expected_hb"][0])
        anemic_probability = float(torch.softmax(outputs["classification_logits"], dim=1)[0, 1])
        severity_index = int(torch.sigmoid(outputs["severity_logits"])[0].gt(0.5).sum())

    result = {
        "frame_detected": True,
        "detection_method": best_frame["detection_method"],
        "passed_qc": True,
        "metrics": quality,
        "hb_gdl": round(predicted_hb, 2),
        "anemic": anemic_probability >= 0.5,
        "anemic_probability": round(anemic_probability, 4),
        "severity": SEVERITY_ORDER[severity_index],
        "severity_caveat": "Akurasi severity masih rendah (Cohen kappa 0.211), interpretasikan dengan hati-hati.",
    }
    result["handcrafted_features"] = {key: round(value, 4) for key, value in handcrafted_values.items()}
    if include_stages:
        result["stage_images"] = {
            "raw_frame": _encode_png_base64(raw_rgb),
            "landmarks": _encode_png_base64(_draw_landmarks_overlay(raw_rgb, landmarks_px)),
            "roi_segmented": _encode_png_base64(_draw_mask_overlay(rgb, mask)),
            "illumination_normalized": _encode_png_base64(normalized_rgb),
            "biomarker_heatmap": _encode_png_base64(_biomarker_heatmap(normalized_rgb, valid_mask)),
        }
    return result
