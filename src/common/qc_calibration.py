"""Kalibrasi gate quality control inference terhadap distribusi data training.

Prinsip kalibrasi adalah foto training merupakan contoh foto yang baik,
sehingga gate quality control di inference tidak boleh menolaknya, sementara
foto yang jelek atau salah objek tetap harus ditolak. Modul ini menyediakan
fungsi untuk mengukur metrik quality control pada set positif (input mentah
data training) dan set negatif (degradasi sintetis, objek bukan tubuh, dan
foto situs lain) lewat fungsi pengukuran yang persis sama dengan jalur
inference, lalu menurunkan batas gate dari persentil set positif dan
mengevaluasi daya pisahnya terhadap set negatif.

Fungsi pengukuran tiap situs berada di app/predict*.py (_measure) dan
disuntikkan oleh notebook driver, agar paket src tidak bergantung pada modul
app.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.common import qc

NEGATIVE_CATEGORIES = (
    "blur",
    "dark",
    "overexposed",
    "tiny",
    "corner_crop",
    "flat_beige",
    "wood_texture",
    "noise",
)
"""Kategori set negatif sintetis yang dibangun dari foto positif setiap situs."""


def load_rgb(path: str | Path, exif_transpose: bool = True) -> np.ndarray:
    """Baca foto sebagai RGB uint8 dengan loader yang sama seperti endpoint inference.

    Konjungtiva dan nail memakai PIL dengan koreksi orientasi EXIF, sedangkan
    palm memakai cv2 tanpa koreksi EXIF, sehingga perilakunya dipilih lewat
    argumen exif_transpose.
    """
    if exif_transpose:
        with Image.open(path) as image:
            return np.array(ImageOps.exif_transpose(image).convert("RGB"))
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Gagal membaca citra: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _flatten_metrics(metrics: dict | None) -> dict:
    """Ambil metrik numerik saja dari hasil pengukuran, buang passed dan reasons."""
    if metrics is None:
        return {"segmentation_failed": True}
    flat = {
        key: value
        for key, value in metrics.items()
        if key not in {"passed", "reasons"} and (value is None or np.isscalar(value))
    }
    flat["segmentation_failed"] = False
    return flat


def measure_images(
    images: list[tuple[str, np.ndarray]] | list[tuple[str, Callable[[], np.ndarray]]],
    measure_fn: Callable[[np.ndarray], dict | None],
    source: str,
) -> pd.DataFrame:
    """Ukur metrik quality control pada sekumpulan citra lewat fungsi pengukuran situs.

    images berisi pasangan (uid, citra RGB) atau (uid, fungsi pemuat citra)
    agar set besar tidak perlu dimuat seluruhnya ke memori. measure_fn
    menerima citra RGB dan mengembalikan dict metrik, atau None bila
    segmentasi gagal total. Galat per citra dicatat pada kolom error, bukan
    menghentikan seluruh pengukuran.
    """
    records = []
    for uid, image in images:
        record = {"uid": uid, "source": source, "error": None}
        try:
            rgb = image() if callable(image) else image
            record.update(_flatten_metrics(measure_fn(rgb)))
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
    return pd.DataFrame(records)


def make_negative_variants(rgb: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Bangun varian negatif sintetis dari satu foto positif.

    Varian mencakup blur berat, gelap, overexposed, objek terlalu kecil di
    frame besar, crop pojok yang umumnya tidak memuat target, serta tiga citra
    bukan tubuh berukuran sama: bidang krem polos menyerupai dinding, tekstur
    serat kayu, dan derau acak.
    """
    height, width = rgb.shape[:2]
    short_side = min(height, width)
    sigma = max(short_side * 0.02, 3.0)
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma)
    dark = (rgb.astype(np.float32) * 0.15).astype(np.uint8)
    overexposed = np.clip(rgb.astype(np.float32) * 2.5 + 90, 0, 255).astype(np.uint8)

    small = cv2.resize(rgb, (max(width // 8, 8), max(height // 8, 8)), interpolation=cv2.INTER_AREA)
    tiny = np.full_like(rgb, 30)
    tiny[:small.shape[0], :small.shape[1]] = small

    corner = rgb[: height // 4, : width // 4]
    corner_crop = cv2.resize(corner, (width, height), interpolation=cv2.INTER_LINEAR)

    beige = np.empty_like(rgb)
    beige[:] = np.array([226, 205, 178], dtype=np.uint8)
    beige = np.clip(beige.astype(np.int16) + rng.normal(0, 4, rgb.shape), 0, 255).astype(np.uint8)

    grain = np.sin(np.linspace(0, 60, width)[None, :] + rng.normal(0, 0.6, (height, 1))) * 25
    wood = np.stack([150 + grain, 95 + grain * 0.7, 50 + grain * 0.4], axis=2)
    wood = np.clip(wood + rng.normal(0, 6, rgb.shape), 0, 255).astype(np.uint8)

    noise = rng.integers(0, 256, rgb.shape, dtype=np.uint8)
    return {
        "blur": blurred,
        "dark": dark,
        "overexposed": overexposed,
        "tiny": tiny,
        "corner_crop": corner_crop,
        "flat_beige": beige,
        "wood_texture": wood,
        "noise": noise,
    }


def measure_negatives(
    positives: list[tuple[str, Callable[[], np.ndarray]]],
    measure_fn: Callable[[np.ndarray], dict | None],
    seed: int = 42,
) -> pd.DataFrame:
    """Ukur metrik quality control seluruh varian negatif dari sekumpulan foto positif."""
    rng = np.random.default_rng(seed)
    frames = []
    for uid, loader in positives:
        variants = make_negative_variants(loader(), rng)
        for category, variant in variants.items():
            frames.append(measure_images([(uid, variant)], measure_fn, source=f"neg_{category}"))
    return pd.concat(frames, ignore_index=True)


def evaluate_policy(frame: pd.DataFrame, gates: dict) -> pd.DataFrame:
    """Terapkan gate pada setiap baris dan tambahkan kolom passed dan reasons.

    Baris yang gagal segmentasi total atau galat pengukuran dianggap ditolak,
    karena di inference keduanya juga tidak menghasilkan prediksi.
    """
    evaluated = frame.copy()
    reasons_column = []
    for _, row in evaluated.iterrows():
        if pd.notna(row.get("error")):
            reasons_column.append(["measure_error"])
            continue
        if bool(row.get("segmentation_failed", False)):
            reasons_column.append(["segmentation_failed"])
            continue
        metrics = {key: (None if pd.isna(value) else value) for key, value in row.items()}
        reasons_column.append(qc.apply_gates(metrics, gates))
    evaluated["reasons"] = reasons_column
    evaluated["passed"] = evaluated["reasons"].map(len) == 0
    return evaluated


def summarize_policy(evaluated: pd.DataFrame) -> pd.DataFrame:
    """Ringkas tingkat lolos per sumber data dan frekuensi tiap alasan penolakan."""
    rows = []
    for source, group in evaluated.groupby("source"):
        reason_counts = pd.Series([r for reasons in group["reasons"] for r in reasons]).value_counts()
        rows.append({
            "source": source,
            "n": len(group),
            "pass_rate": round(float(group["passed"].mean()), 4),
            "top_reasons": ", ".join(f"{reason} {count}" for reason, count in reason_counts.head(4).items()),
        })
    return pd.DataFrame(rows)


def _resolve_bound(values: pd.Series, spec, side: str) -> float | None:
    """Hitung satu sisi batas gate dari spesifikasi aturan.

    Spesifikasi None berarti sisi tersebut tidak di-gate, angka berarti batas
    tetap yang ditetapkan manual, dan tuple ("pct", persentil, margin) berarti
    persentil set positif yang dilonggarkan dengan margin relatif, dikurangi
    untuk batas bawah dan ditambah untuk batas atas.
    """
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return float(spec)
    _, percentile, margin = spec
    anchor = float(np.percentile(values.dropna().astype(float), percentile))
    return anchor * (1 - margin) if side == "min" else anchor * (1 + margin)


def derive_gates(positives: pd.DataFrame, rules: dict) -> dict:
    """Turunkan batas gate seluruh metrik satu situs dari aturan dan set positif.

    rules berbentuk {metrik: {"min": spesifikasi, "max": spesifikasi}}, lihat
    _resolve_bound untuk format spesifikasi. Nilai dibulatkan empat angka
    penting agar configs/qc_thresholds.json mudah dibaca.
    """
    gates = {}
    for metric, rule in rules.items():
        bound = {}
        for side in ("min", "max"):
            value = _resolve_bound(positives[metric], rule.get(side), side)
            bound[side] = None if value is None else float(f"{value:.4g}")
        gates[metric] = bound
    return gates


def write_thresholds(policy_by_site: dict, provenance: dict, path: Path = qc.QC_THRESHOLDS_PATH) -> None:
    """Tulis configs/qc_thresholds.json beserta catatan asal-usul kalibrasi."""
    payload = {
        "version": f"calibrated-{date.today().isoformat()}",
        "calibration": provenance,
        "sites": {site: {"gates": gates} for site, gates in policy_by_site.items()},
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    qc.load_site_gates.cache_clear()
