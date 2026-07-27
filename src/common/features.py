"""Dual-path feature extraction.

Menyediakan jalur fitur warna hand-crafted yang interpretable (statistik kanal
RGB, HSV, CIELAB, rasio terkait hemoglobin, erythema index, entropy, dan fitur
tekstur) serta jalur deep embedding berbasis backbone ringan dengan modul
channel spatial attention. Kedua jalur difusikan bersama variabel demografi dan
site token memakai modul fusion attention.

Fitur hand-crafted dihitung pada pixel valid hasil normalisasi Stage 2 agar
konsisten dengan pipeline iluminasi yang sudah divalidasi. Jalur deep embedding
memakai ResNet18 pralatih ImageNet dengan modul channel spatial attention yang
disisipkan setelah blok pertama untuk menonjolkan detail halus seperti pembuluh
darah, mengikuti rasionalisasi pada literatur BPANet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(0)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import ARTIFACTS, OUTPUTS
from src.common.preprocess import normalize_roi

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
EMBEDDING_DIM = 256

HANDCRAFTED_COLUMNS = [
    "mean_r", "mean_g", "mean_b", "mean_r_minus_g",
    "hhr", "hhr_hue", "red_ratio", "erythema_index",
    "mean_h", "std_h", "mean_s", "std_s", "mean_v", "std_v",
    "mean_l", "std_l", "mean_a", "std_a", "mean_bb", "std_bb",
    "g1", "g2", "g3", "g4", "g5",
    "entropy", "brightness",
]


def _texture_features(gray, mask):
    """Hitung fitur tekstur gray level lokal g1 hingga g5 pada jendela 3x3.

    g1 hingga g4 menangkap variasi intensitas terhadap tetangga lokal, sedangkan
    g5 adalah intensitas itu sendiri. Setiap fitur dirata-ratakan pada pixel
    valid untuk menghasilkan satu nilai per citra.
    """
    gray_float = gray.astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    local_min = cv2.erode(gray_float, kernel)
    local_max = cv2.dilate(gray_float, kernel)
    local_mean = cv2.blur(gray_float, (3, 3))
    local_sq_mean = cv2.blur(gray_float ** 2, (3, 3))
    local_variance = np.clip(local_sq_mean - local_mean ** 2, 0, None)
    local_std = np.sqrt(local_variance)

    g1 = gray_float - local_min
    g2 = local_max - gray_float
    g3 = gray_float - local_mean
    g4 = local_std
    g5 = gray_float

    return {
        "g1": float(g1[mask].mean()) if mask.any() else 0.0,
        "g2": float(g2[mask].mean()) if mask.any() else 0.0,
        "g3": float(g3[mask].mean()) if mask.any() else 0.0,
        "g4": float(g4[mask].mean()) if mask.any() else 0.0,
        "g5": float(g5[mask].mean()) if mask.any() else 0.0,
    }


def _entropy(gray, mask) -> float:
    """Hitung entropy grayscale pada pixel valid memakai histogram 256 bin."""
    if not mask.any():
        return 0.0
    values = gray[mask]
    histogram, _ = np.histogram(values, bins=256, range=(0, 255))
    probability = histogram / histogram.sum()
    probability = probability[probability > 0]
    return float(-(probability * np.log2(probability)).sum())


def compute_handcrafted_features(rgb, valid_mask) -> dict:
    """Hitung vektor fitur warna hand-crafted dari satu citra region of interest.

    Fitur mencakup statistik RGB, rasio terkait hemoglobin, statistik HSV dan
    CIELAB, erythema index, tekstur gray level, entropy, dan brightness. Semua
    dihitung hanya pada pixel yang ditandai valid oleh Stage 2.
    """
    mask = valid_mask
    if not mask.any():
        return {column: 0.0 for column in HANDCRAFTED_COLUMNS}

    rgb_float = rgb.astype(np.float32)
    mean_r = float(rgb_float[:, :, 0][mask].mean())
    mean_g = float(rgb_float[:, :, 1][mask].mean())
    mean_b = float(rgb_float[:, :, 2][mask].mean())
    eps = 1e-6

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_normalized = hsv[:, :, 0] / 179.0
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel, a_channel, b_channel = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    features = {
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "mean_r_minus_g": mean_r - mean_g,
        "hhr": mean_r / (mean_g + eps),
        "hhr_hue": float((hue_normalized[mask] > 0.95).mean()),
        "red_ratio": mean_r / (mean_r + mean_g + mean_b + eps),
        "erythema_index": float(np.log10(1.0 / (mean_g / 255.0 + eps))),
        "mean_h": float(hue_normalized[mask].mean()),
        "std_h": float(hue_normalized[mask].std()),
        "mean_s": float(saturation[mask].mean()),
        "std_s": float(saturation[mask].std()),
        "mean_v": float(value[mask].mean()),
        "std_v": float(value[mask].std()),
        "mean_l": float(l_channel[mask].mean()),
        "std_l": float(l_channel[mask].std()),
        "mean_a": float(a_channel[mask].mean()),
        "std_a": float(a_channel[mask].std()),
        "mean_bb": float(b_channel[mask].mean()),
        "std_bb": float(b_channel[mask].std()),
        "entropy": _entropy(gray, mask),
        "brightness": float(gray[mask].mean()),
    }
    features.update(_texture_features(gray, mask))
    return features


def extract_handcrafted_features(manifest: pd.DataFrame) -> pd.DataFrame:
    """Jalankan ekstraksi fitur hand-crafted untuk seluruh baris manifest."""
    records = []
    for _, row in manifest.iterrows():
        normalized = normalize_roi(row["roi_path"])
        features = compute_handcrafted_features(normalized["rgb"], normalized["valid_mask"])
        features["uid"] = row["uid"]
        records.append(features)
    frame = pd.DataFrame(records)
    return frame[["uid"] + HANDCRAFTED_COLUMNS]


class ChannelAttention(nn.Module):
    """Modul channel attention gaya CBAM memakai pooling rata-rata dan maksimum."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.last_gate = None

    def forward(self, x):
        gate = torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        self.last_gate = gate.detach()
        return x * gate


class SpatialAttention(nn.Module):
    """Modul spatial attention gaya CBAM memakai konvolusi pada peta rata-rata dan maksimum."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.last_gate = None

    def forward(self, x):
        average_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat([average_map, max_map], dim=1)))
        self.last_gate = gate.detach()
        return x * gate


class CSABlock(nn.Module):
    """Channel spatial attention, menerapkan channel attention lalu spatial attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


def _split_backbone(backbone_name: str, pretrained: bool):
    """Bangun bagian awal dan bagian lanjutan sebuah backbone pralatih ImageNet.

    Bagian awal adalah titik penyisipan CSA, dipilih pada tahap dangkal jaringan
    agar atensi menonjolkan detail halus seperti pembuluh darah sebelum
    representasi menjadi terlalu abstrak. Backbone yang didukung adalah resnet18
    dan mobilenet_v3_small, mewakili perbandingan model berat lawan model ringan
    yang ramah perangkat edge.
    """
    from torchvision import models

    if backbone_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        early = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        late = nn.Sequential(backbone.layer2, backbone.layer3, backbone.layer4)
        return early, late

    if backbone_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        blocks = list(backbone.features.children())
        split = 4
        early = nn.Sequential(*blocks[:split])
        late = nn.Sequential(*blocks[split:])
        return early, late

    raise ValueError(f"backbone_name tidak dikenal: {backbone_name}")


class EmbeddingBackbone(nn.Module):
    """Backbone pralatih ImageNet dengan modul CSA disisipkan pada tahap dangkal.

    Mendukung resnet18 (representasi lebih kaya, cocok sebagai kandidat utama)
    dan mobilenet_v3_small (lebih ringan, relevan untuk deployment perangkat
    edge). Dimensi channel pada titik penyisipan CSA dan pada keluaran akhir
    backbone dideteksi otomatis lewat forward pass dummy, sehingga tidak
    bergantung pada nilai channel yang di-hardcode per arsitektur.
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        pretrained: bool = True,
        backbone_name: str = "resnet18",
        use_csa: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.use_csa = use_csa
        self.early, self.late = _split_backbone(backbone_name, pretrained)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            early_output = self.early(dummy)
            csa_channels = early_output.shape[1]
            late_output = self.late(early_output)
            final_channels = late_output.shape[1]

        if seed is not None:
            torch.manual_seed(seed)
        self.csa = CSABlock(channels=csa_channels) if use_csa else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(final_channels, embedding_dim)

    def forward(self, x):
        x = self.early(x)
        x = self.csa(x)
        x = self.late(x)
        x = self.pool(x).flatten(1)
        return self.projection(x)


MAX_ROTATION_DEGREES = 15.0
BRIGHTNESS_JITTER = 0.15
CONTRAST_JITTER = 0.15


def _augment_roi(rgb: np.ndarray) -> np.ndarray:
    """Augmentasi ringan untuk fine-tuning end-to-end, dijalankan hanya pada split train.

    Hanya memakai transformasi geometris (flip, rotasi kecil) dan jitter
    brightness/contrast ringan. Sengaja tidak memakai hue/saturation/color
    jitter karena warna kulit (pallor, erythema index) adalah sinyal diagnostik
    utama yang ingin dipelajari model, bukan noise yang boleh dirusak.
    """
    if np.random.rand() < 0.5:
        rgb = np.ascontiguousarray(rgb[:, ::-1, :])

    angle = np.random.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES)
    height, width = rgb.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rgb = cv2.warpAffine(
        rgb, rotation_matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )

    brightness_factor = 1.0 + np.random.uniform(-BRIGHTNESS_JITTER, BRIGHTNESS_JITTER)
    contrast_factor = 1.0 + np.random.uniform(-CONTRAST_JITTER, CONTRAST_JITTER)
    rgb_float = rgb.astype(np.float32) * brightness_factor
    mean_intensity = rgb_float.mean()
    rgb_float = (rgb_float - mean_intensity) * contrast_factor + mean_intensity
    return np.clip(rgb_float, 0.0, 255.0).astype(np.uint8)


def load_normalized_resized(roi_path: str, size: int = 224) -> np.ndarray:
    """Muat dan normalisasi satu citra ROI, lalu ubah ukuran, tanpa augmentasi.

    Hasil fungsi ini deterministik terhadap roi_path dan size, sehingga aman
    di-cache lintas epoch lewat precompute_normalized_images, mengingat
    normalize_roi (termasuk CLAHE) adalah langkah paling mahal pada pipeline
    pemuatan citra.
    """
    normalized = normalize_roi(roi_path)
    return cv2.resize(normalized["rgb"], (size, size), interpolation=cv2.INTER_AREA)


def precompute_normalized_images(manifest: pd.DataFrame, size: int = 224) -> dict:
    """Precompute dan cache citra ROI ternormalisasi untuk seluruh baris manifest.

    Dipanggil sekali sebelum loop k-fold pada fine-tuning end-to-end, sehingga
    normalize_roi (mahal karena CLAHE) tidak dihitung ulang pada setiap epoch
    dan setiap fold. Cache dibangun di proses utama sebelum DataLoader
    membuat worker, sehingga worker yang di-fork mewarisi cache ini tanpa
    salinan tambahan (copy-on-write) alih-alih memuat ulang dari disk.
    """
    return {
        row["uid"]: load_normalized_resized(row["roi_path"], size=size)
        for _, row in manifest.iterrows()
    }


def load_roi_tensor(
    roi_path: str, size: int = 224, augment: bool = False, cached_rgb: np.ndarray | None = None,
) -> torch.Tensor:
    """Muat citra region of interest, ubah ukuran, dan normalisasi gaya ImageNet.

    Dipakai bersama oleh ROIImageDataset dan dataset pelatihan end-to-end agar
    logika pemuatan citra tidak terduplikasi. Augmentasi hanya diterapkan bila
    augment=True, dipakai pada split train saat fine-tuning end-to-end. Bila
    cached_rgb diisi (dari precompute_normalized_images), langkah normalize_roi
    dan resize dilewati dan hasil cache dipakai langsung sebelum augmentasi.
    """
    if cached_rgb is not None:
        rgb = cached_rgb
    else:
        rgb = load_normalized_resized(roi_path, size=size)
    if augment:
        rgb = _augment_roi(rgb)
    rgb_float = rgb.astype(np.float32) / 255.0
    rgb_float = (rgb_float - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb_float.transpose(2, 0, 1)).float()


def freeze_backbone_body(model: "EmbeddingBackbone") -> None:
    """Bekukan bobot badan backbone (early dan late), sisakan CSA dan proyeksi trainable.

    Mengikuti pendekatan transfer learning standar, badan ResNet atau MobileNet
    pralatih ImageNet dipertahankan tetap, sedangkan modul CSA dan proyeksi
    dilatih bersama fusion attention dan head lewat backpropagation penuh.
    """
    for parameter in model.early.parameters():
        parameter.requires_grad = False
    for parameter in model.late.parameters():
        parameter.requires_grad = False


class ROIImageDataset(Dataset):
    """Dataset citra region of interest ternormalisasi untuk jalur deep embedding."""

    def __init__(self, frame: pd.DataFrame, size: int = 224):
        self.rows = frame.reset_index(drop=True)
        self.size = size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        tensor = load_roi_tensor(row["roi_path"], size=self.size)
        return tensor, row["uid"]


def extract_deep_embeddings(
    manifest: pd.DataFrame,
    model: nn.Module | None = None,
    size: int = 224,
    batch_size: int = 32,
    device: str | None = None,
):
    """Jalankan ekstraksi deep embedding untuk seluruh baris manifest.

    Mengembalikan array embedding dengan urutan baris mengikuti manifest serta
    daftar uid yang berkorespondensi untuk verifikasi kesesuaian urutan.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model or EmbeddingBackbone()
    model = model.to(device)
    model.eval()

    dataset = ROIImageDataset(manifest, size=size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    embeddings = []
    uids = []
    with torch.no_grad():
        for images, batch_uids in loader:
            images = images.to(device)
            output = model(images)
            embeddings.append(output.cpu().numpy())
            uids.extend(batch_uids)
    return np.concatenate(embeddings, axis=0), uids


class FusionAttention(nn.Module):
    """Modul fusion attention yang menggabungkan fitur dari berbagai sumber.

    Mengikuti rasionalisasi pada literatur BPANet, atensi berbentuk vektor Q
    dikalikan matriks key yang dapat dipelajari, dilewatkan sigmoid, lalu
    dikalikan matriks value yang dapat dipelajari untuk menghasilkan fitur
    fusi dengan dimensi yang sama dengan fitur masukan.
    """

    def __init__(self, feature_dim: int, attention_dim: int = 64):
        super().__init__()
        self.key = nn.Linear(feature_dim, attention_dim, bias=False)
        self.value = nn.Linear(attention_dim, feature_dim, bias=False)

    def forward(self, x):
        attention = torch.sigmoid(self.key(x))
        return self.value(attention)


def compute_fusion_stats(handcrafted: pd.DataFrame, manifest: pd.DataFrame, site_categories: list | None = None) -> dict:
    """Hitung statistik standardisasi dan kosakata site tetap dari satu subset manifest.

    Statistik ini wajib dihitung hanya dari fold atau split train agar tidak
    terjadi kebocoran data ke fold atau split validasi. Kosakata site ditetapkan
    eksplisit agar lebar one-hot encoding tetap konsisten meski satu subset
    kebetulan tidak memuat seluruh kategori site.
    """
    merged = manifest[["uid", "site"]].merge(handcrafted, on="uid", how="left")
    handcrafted_values = merged[HANDCRAFTED_COLUMNS].to_numpy(dtype=np.float32)
    mean = handcrafted_values.mean(axis=0, keepdims=True)
    std = handcrafted_values.std(axis=0, keepdims=True) + 1e-6
    categories = site_categories or sorted(manifest["site"].unique().tolist())
    return {"mean": mean, "std": std, "site_categories": categories}


def build_fusion_input(
    handcrafted: pd.DataFrame,
    deep_embeddings: np.ndarray,
    manifest: pd.DataFrame,
    stats: dict | None = None,
    use_handcrafted: bool = True,
    use_deep: bool = True,
    use_demographics: bool = True,
    use_site_token: bool = True,
    extra_columns: list[str] | None = None,
) -> np.ndarray:
    """Susun vektor fusi dari fitur hand-crafted, deep embedding, demografi, dan site token.

    Fitur hand-crafted distandarisasi memakai stats yang diberikan, umur
    dinormalisasi ke rentang wajar, gender dikodekan biner, dan site dikodekan
    sebagai one-hot dengan kosakata tetap dari stats. Bila stats tidak diisi,
    dihitung otomatis dari manifest yang diberikan, cocok untuk eksplorasi pada
    seluruh dataset. Untuk pelatihan dengan validasi silang, stats harus selalu
    dihitung dari fold train saja lewat compute_fusion_stats.

    Parameter use_handcrafted dan use_deep memungkinkan penyusunan vektor fitur
    untuk konfigurasi jalur tunggal (Path A saja atau Path B saja) memakai
    perlakuan demografi dan site token yang identik dengan konfigurasi fusi
    penuh, sehingga perbandingan antar konfigurasi tetap adil. Parameter
    use_demographics dan use_site_token memungkinkan ablation komponen umur dan
    gender, atau site token, secara terpisah.

    Parameter extra_columns bersifat opsional dan situs-agnostik, dipakai untuk
    menambahkan kolom kategorikal/biner khusus situs dari manifest (misalnya
    polish_flag pada situs nail) tanpa mem-fork fungsi ini. Kosong secara
    default sehingga tidak mengubah perilaku situs yang sudah ada.
    """
    stats = stats or compute_fusion_stats(handcrafted, manifest)
    extra_columns = extra_columns or []
    merged = manifest[["uid", "age_years", "gender", "site"] + extra_columns].merge(
        handcrafted, on="uid", how="left"
    )

    blocks = []
    if use_handcrafted:
        handcrafted_values = merged[HANDCRAFTED_COLUMNS].to_numpy(dtype=np.float32)
        blocks.append((handcrafted_values - stats["mean"]) / stats["std"])
    if use_deep:
        blocks.append(deep_embeddings.astype(np.float32))
    if use_demographics:
        age_normalized = (merged["age_years"].fillna(0).to_numpy(dtype=np.float32) / 100.0).reshape(-1, 1)
        gender_binary = (merged["gender"] == "M").astype(np.float32).to_numpy().reshape(-1, 1)
        blocks.extend([age_normalized, gender_binary])
    if use_site_token:
        site_categorical = pd.Categorical(merged["site"], categories=stats["site_categories"])
        site_dummies = pd.get_dummies(site_categorical).to_numpy(dtype=np.float32)
        blocks.append(site_dummies)
    if extra_columns:
        extra_values = merged[extra_columns].astype(np.float32).fillna(0.0).to_numpy()
        blocks.append(extra_values)

    return np.concatenate(blocks, axis=1)


def save_features(
    handcrafted: pd.DataFrame,
    deep_embeddings: np.ndarray,
    uids: list,
    output_dir: Path | None = None,
) -> dict:
    """Simpan fitur hand-crafted dan deep embedding ke folder output yang ditentukan.

    Bila output_dir tidak diisi, fitur disimpan ke folder outputs datar. Notebook
    situs sebaiknya selalu mengisi output_dir dengan folder bernamespace situs
    agar tidak menimpa fitur milik situs lain.
    """
    target = output_dir or OUTPUTS
    target.mkdir(parents=True, exist_ok=True)
    handcrafted_path = target / "handcrafted_features.csv"
    embeddings_path = target / "deep_embeddings.npy"
    uids_path = target / "deep_embeddings_uids.csv"

    handcrafted.to_csv(handcrafted_path, index=False)
    np.save(embeddings_path, deep_embeddings)
    pd.DataFrame({"uid": uids}).to_csv(uids_path, index=False)

    return {
        "handcrafted": str(handcrafted_path),
        "embeddings": str(embeddings_path),
        "uids": str(uids_path),
    }
