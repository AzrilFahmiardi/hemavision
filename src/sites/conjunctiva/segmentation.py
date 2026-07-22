"""Segmentation front-end untuk situs konjungtiva.

Melatih U-Net pada foto mata utuh Eyes-Defy untuk memprediksi mask konjungtiva
palpebral. Segmentasi ini menjadi jembatan dari citra mentah menuju region of
interest pada mode capture di lapangan, sekaligus komponen visual pipeline.

Foto mentah Eyes-Defy tersimpan dengan orientasi EXIF sehingga dimuat memakai
koreksi EXIF agar sejajar dengan mask yang berorientasi portrait. Mask target
diturunkan dari file mask memakai loader format aware pada modul quality control.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.paths import ARTIFACTS, OUTPUTS
from src.common.qc import load_roi

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_raw_rgb(path: str) -> np.ndarray:
    """Muat foto mata utuh dengan koreksi orientasi EXIF sebagai array RGB."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.array(image)


def load_binary_mask(path: str) -> np.ndarray:
    """Turunkan mask biner konjungtiva dari file mask memakai loader format aware."""
    _, mask = load_roi(path)
    return mask.astype(np.uint8)


def get_transforms(size: int, train: bool):
    """Bangun pipeline augmentasi dan normalisasi untuk citra dan mask."""
    steps = [A.Resize(size, size)]
    if train:
        steps += [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ]
    steps += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(steps)


class SegmentationDataset(Dataset):
    """Pasangan foto mata utuh dan mask palpebral untuk pelatihan segmentasi."""

    def __init__(self, frame, size: int = 320, train: bool = False):
        self.rows = frame.reset_index(drop=True)
        self.transform = get_transforms(size, train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image = load_raw_rgb(row["raw_path"])
        mask = load_binary_mask(row["mask_path"])
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented["image"]
        mask_tensor = augmented["mask"].unsqueeze(0).float()
        return image_tensor, mask_tensor


def build_datasets(manifest, size: int = 320):
    """Bangun dataset train dan val dari baris Eyes-Defy pada manifest."""
    eyes = manifest[manifest["dataset"] == "eyes_defy"].copy()
    train_rows = eyes[eyes["split"] == "train"]
    val_rows = eyes[eyes["split"].isin(["val", "test"])]
    train_ds = SegmentationDataset(train_rows, size=size, train=True)
    val_ds = SegmentationDataset(val_rows, size=size, train=False)
    return train_ds, val_ds


def build_model(encoder: str = "resnet34"):
    """Bangun U-Net dengan encoder pralatih ImageNet, fallback tanpa bobot pralatih."""
    try:
        return smp.Unet(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=1)
    except Exception as error:
        print(f"Gagal memuat bobot ImageNet ({error}), memakai inisialisasi acak.")
        return smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)


def _dice_iou(logits, target, eps: float = 1e-6):
    """Hitung Dice dan IoU dari logit prediksi terhadap mask target."""
    probability = torch.sigmoid(logits)
    prediction = (probability > 0.5).float()
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    union = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    iou = (intersection + eps) / (union - intersection + eps)
    return dice.mean().item(), iou.mean().item()


def train_unet(
    manifest,
    size: int = 320,
    epochs: int = 25,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    encoder: str = "resnet34",
    device: str | None = None,
    artifacts_dir: Path | None = None,
    output_dir: Path | None = None,
):
    """Latih U-Net segmentasi dan simpan checkpoint serta metrik ke disk.

    Bila artifacts_dir atau output_dir tidak diisi, keduanya jatuh ke folder
    datar ARTIFACTS dan OUTPUTS. Notebook situs sebaiknya selalu mengisi kedua
    parameter dengan folder bernamespace situs.

    Mengembalikan model terlatih, riwayat pelatihan, dan metrik validasi terbaik.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    artifacts_dir = artifacts_dir or ARTIFACTS
    output_dir = output_dir or OUTPUTS
    train_ds, val_ds = build_datasets(manifest, size=size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(encoder).to(device)
    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_dice": [], "val_iou": []}
    best_dice = 0.0
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_dir / "segmentation_unet.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = dice_loss(logits, masks) + bce_loss(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
        epoch_loss /= len(train_ds)

        model.eval()
        dice_values, iou_values = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                dice, iou = _dice_iou(logits, masks)
                dice_values.append(dice)
                iou_values.append(iou)
        val_dice = float(np.mean(dice_values)) if dice_values else 0.0
        val_iou = float(np.mean(iou_values)) if iou_values else 0.0
        history["train_loss"].append(epoch_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        print(f"epoch {epoch:02d} train_loss {epoch_loss:.4f} val_dice {val_dice:.4f} val_iou {val_iou:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)

    metrics = {"best_val_dice": best_dice, "final_val_iou": history["val_iou"][-1], "epochs": epochs, "size": size}
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "segmentation_metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    return {"model": model, "history": history, "metrics": metrics, "checkpoint": str(checkpoint_path)}


def denormalize(image_tensor) -> np.ndarray:
    """Kembalikan tensor citra ternormalisasi menjadi array RGB untuk visualisasi."""
    array = image_tensor.cpu().numpy().transpose(1, 2, 0)
    array = array * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(array, 0, 1)


def save_overlays(model, dataset, count: int = 4, device: str | None = None, output_dir: Path | None = None):
    """Simpan contoh overlay prediksi mask pada citra validasi ke folder output.

    Bila output_dir tidak diisi, overlay disimpan ke folder outputs datar.
    """
    import matplotlib.pyplot as plt

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = output_dir or OUTPUTS
    model.eval()
    count = min(count, len(dataset))
    fig, axes = plt.subplots(count, 3, figsize=(9, 3 * count))
    if count == 1:
        axes = axes[None, :]
    with torch.no_grad():
        for i in range(count):
            image_tensor, mask_tensor = dataset[i]
            logits = model(image_tensor.unsqueeze(0).to(device))
            prediction = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5)
            rgb = denormalize(image_tensor)
            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title("image")
            axes[i, 1].imshow(mask_tensor[0].numpy(), cmap="gray")
            axes[i, 1].set_title("ground truth")
            axes[i, 2].imshow(rgb)
            axes[i, 2].imshow(prediction, cmap="Reds", alpha=0.4)
            axes[i, 2].set_title("prediction overlay")
            for ax in axes[i]:
                ax.axis("off")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "segmentation_overlays.png"
    plt.savefig(output_path, dpi=110)
    plt.close(fig)
    return str(output_path)
