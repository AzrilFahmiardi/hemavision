"""Multi-task model and heads.

Mendefinisikan arsitektur multi-task yang menerima vektor fusi (hasil dual-path
feature extraction, demografi, dan site token), memprosesnya lewat modul fusion
attention dan trunk bersama, lalu memprediksi tiga tugas sekaligus, yakni
regresi hemoglobin, klasifikasi biner anemia, dan estimasi severity ordinal.

Head regresi hemoglobin mengikuti pendekatan distribusi atas bin (bukan regresi
langsung), sehingga ketidakpastian dan struktur kelas dapat dimanfaatkan lewat
Dual Loss. Head severity ordinal memakai formulasi CORN yang menjaga konsistensi
urutan kelas keparahan.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.features import FusionAttention

HB_RANGE = (0.0, 24.0)
N_BINS = 48
SEVERITY_CLASSES = 4
SEVERITY_ORDER = ["Non-Anemic", "Mild", "Moderate", "Severe"]


class FocalLoss(nn.Module):
    """Focal loss multi-kelas untuk menangani ketidakseimbangan kelas.

    Dipakai baik untuk klasifikasi biner anemia (dua kelas) maupun untuk
    komponen klasifikasi pada Dual Loss regresi hemoglobin (klasifikasi bin).
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probability = F.log_softmax(logits, dim=-1)
        probability = log_probability.exp()
        target_log_probability = log_probability.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probability = probability.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - target_probability).clamp(min=0.0) ** self.gamma
        loss = -focal_weight * target_log_probability
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[targets]
        return loss.mean()


class DualLoss(nn.Module):
    """Dual Loss untuk regresi hemoglobin, menggabungkan MSE dan focal loss atas bin.

    Mengikuti rasionalisasi BPANet, MSE dihitung pada nilai ekspektasi hasil
    distribusi bin, sedangkan komponen klasifikasi menangkap ketidakseimbangan
    dan struktur ordinal kadar hemoglobin lewat focal loss atas indeks bin.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.focal = FocalLoss(gamma=gamma)

    def forward(self, regression_logits, expected_hb, target_hb, bin_index) -> torch.Tensor:
        mse_loss = self.mse(expected_hb, target_hb)
        focal_loss = self.focal(regression_logits, bin_index)
        return mse_loss + self.alpha * focal_loss


class CornLoss(nn.Module):
    """CORN ordinal loss untuk klasifikasi severity yang menjaga konsistensi urutan.

    Model diharapkan mengeluarkan K minus satu logit untuk problem ordinal
    dengan K kelas. Logit ke-k dilatih secara kondisional hanya pada sampel
    dengan label asli lebih besar atau sama dengan k, mengikuti formulasi CORN
    (Shi, Cao, dan Raschka).
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_thresholds = logits.shape[1]
        total_loss = logits.new_zeros(())
        total_count = 0
        for threshold in range(num_thresholds):
            conditional_mask = targets >= threshold
            count = int(conditional_mask.sum().item())
            if count == 0:
                continue
            binary_target = (targets[conditional_mask] > threshold).float()
            binary_logit = logits[conditional_mask, threshold]
            total_loss = total_loss + F.binary_cross_entropy_with_logits(
                binary_logit, binary_target, reduction="sum"
            )
            total_count += count
        if total_count == 0:
            return logits.new_zeros(())
        return total_loss / total_count

    @staticmethod
    def predict_rank(logits: torch.Tensor) -> torch.Tensor:
        """Turunkan kelas ordinal prediksi dari jumlah ambang yang terlampaui."""
        probability = torch.sigmoid(logits)
        return (probability > 0.5).sum(dim=1)


class HemavisionModel(nn.Module):
    """Model multi-task dual-path untuk skrining anemia konjungtiva.

    Vektor fusi diproses lewat modul fusion attention, lalu trunk bersama,
    sebelum dipecah menjadi tiga head. Dimensi vektor fusi ditentukan oleh
    konfigurasi jalur yang dipakai (Path A saja, Path B saja, atau keduanya),
    sehingga input_dim harus disesuaikan dengan bentuk data yang sebenarnya.
    """

    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 64,
        trunk_dim: int = 128,
        n_bins: int = N_BINS,
        hb_range: tuple[float, float] = HB_RANGE,
        severity_classes: int = SEVERITY_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fusion_attention = FusionAttention(feature_dim=input_dim, attention_dim=attention_dim)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, trunk_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(trunk_dim, trunk_dim),
            nn.ReLU(inplace=True),
        )
        self.regression_head = nn.Linear(trunk_dim, n_bins)
        self.classification_head = nn.Linear(trunk_dim, 2)
        self.severity_head = nn.Linear(trunk_dim, severity_classes - 1)

        bin_edges = torch.linspace(hb_range[0], hb_range[1], n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.register_buffer("bin_centers", bin_centers)
        self.n_bins = n_bins
        self.hb_min = hb_range[0]
        self.bin_width = (hb_range[1] - hb_range[0]) / n_bins

    def forward(self, x: torch.Tensor) -> dict:
        fused = self.fusion_attention(x)
        trunk_output = self.trunk(fused)
        regression_logits = self.regression_head(trunk_output)
        classification_logits = self.classification_head(trunk_output)
        severity_logits = self.severity_head(trunk_output)
        probability = torch.softmax(regression_logits, dim=-1)
        expected_hb = (probability * self.bin_centers.unsqueeze(0)).sum(dim=-1)
        return {
            "regression_logits": regression_logits,
            "expected_hb": expected_hb,
            "classification_logits": classification_logits,
            "severity_logits": severity_logits,
        }

    def hb_to_bin_index(self, hb_values: torch.Tensor) -> torch.Tensor:
        """Konversi nilai hemoglobin menjadi indeks bin untuk target Dual Loss."""
        index = ((hb_values - self.hb_min) / self.bin_width).long()
        return index.clamp(0, self.n_bins - 1)
