"""Training utilities untuk model multi-task.

Menyediakan validasi silang k-fold berbasis pasien untuk model multi-task
dual-path. Standardisasi fitur hand-crafted dan kosakata site dihitung ulang
pada setiap fold hanya dari bagian train agar tidak terjadi kebocoran data ke
bagian validasi. Fungsi ini menerima konfigurasi jalur (Path A saja, Path B
saja, atau keduanya) sehingga dapat dipakai ulang untuk membandingkan beberapa
konfigurasi model dengan protokol yang identik.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.features import build_fusion_input, compute_fusion_stats
from src.common.models import CornLoss, DualLoss, FocalLoss, HemavisionModel, SEVERITY_ORDER


def _select_embeddings(embedding_uids, embeddings, target_uids) -> np.ndarray:
    """Ambil baris deep embedding sesuai urutan uid pada subset manifest tertentu."""
    lookup = {uid: row for uid, row in zip(embedding_uids, embeddings)}
    return np.stack([lookup[uid] for uid in target_uids])


def _severity_to_ordinal(series: pd.Series) -> np.ndarray:
    """Konversi label severity menjadi indeks ordinal, minus satu bila tidak ada label."""
    mapping = {name: index for index, name in enumerate(SEVERITY_ORDER)}
    return series.map(lambda value: mapping.get(value, -1)).to_numpy()


def _class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Hitung bobot kelas berbanding terbalik dengan frekuensi, untuk focal loss."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _iterate_batches(n_rows: int, batch_size: int, rng: np.random.Generator):
    """Hasilkan indeks batch teracak untuk satu epoch pelatihan full-batch kecil."""
    order = rng.permutation(n_rows)
    for start in range(0, n_rows, batch_size):
        yield order[start:start + batch_size]


def run_kfold(
    manifest: pd.DataFrame,
    handcrafted: pd.DataFrame,
    deep_embeddings: np.ndarray | None,
    embedding_uids: list | None,
    n_splits: int = 5,
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 0.5),
    use_handcrafted: bool = True,
    use_deep: bool = True,
    device: str | None = None,
    seed: int = 42,
) -> dict:
    """Latih dan evaluasi model multi-task memakai validasi silang k-fold.

    Manifest harus sudah memiliki kolom fold dari assign_kfold. Mengembalikan
    prediksi out-of-fold untuk seluruh sampel, metrik ringkas per fold, dan
    daftar model terlatih per fold untuk keperluan penyimpanan checkpoint.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weight_regression, weight_classification, weight_severity = loss_weights

    severity_ordinal = _severity_to_ordinal(manifest["severity"])
    has_severity = severity_ordinal >= 0
    site_categories = sorted(manifest["site"].unique().tolist())

    oof_records = []
    fold_metrics = []
    models = []

    for fold in sorted(manifest["fold"].unique()):
        train_mask = manifest["fold"] != fold
        val_mask = manifest["fold"] == fold
        train_manifest = manifest[train_mask].reset_index(drop=True)
        val_manifest = manifest[val_mask].reset_index(drop=True)

        stats = compute_fusion_stats(handcrafted, train_manifest, site_categories=site_categories)

        train_deep = _select_embeddings(embedding_uids, deep_embeddings, train_manifest["uid"]) if use_deep else None
        val_deep = _select_embeddings(embedding_uids, deep_embeddings, val_manifest["uid"]) if use_deep else None

        train_features = build_fusion_input(
            handcrafted, train_deep, train_manifest, stats=stats,
            use_handcrafted=use_handcrafted, use_deep=use_deep,
        )
        val_features = build_fusion_input(
            handcrafted, val_deep, val_manifest, stats=stats,
            use_handcrafted=use_handcrafted, use_deep=use_deep,
        )

        train_hb = train_manifest["hb_gdl"].to_numpy(dtype=np.float32)
        val_hb = val_manifest["hb_gdl"].to_numpy(dtype=np.float32)
        train_anemic = train_manifest["anemic"].to_numpy(dtype=np.int64)
        val_anemic = val_manifest["anemic"].to_numpy(dtype=np.int64)
        train_severity = severity_ordinal[train_mask.to_numpy()]
        val_severity = severity_ordinal[val_mask.to_numpy()]
        train_has_severity = has_severity[train_mask.to_numpy()]
        val_has_severity = has_severity[val_mask.to_numpy()]

        model = HemavisionModel(input_dim=train_features.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        dual_loss_fn = DualLoss()
        classification_alpha = _class_weights(train_anemic, num_classes=2)
        classification_loss_fn = FocalLoss(alpha=classification_alpha)
        severity_loss_fn = CornLoss()

        train_features_tensor = torch.tensor(train_features, dtype=torch.float32, device=device)
        train_hb_tensor = torch.tensor(train_hb, dtype=torch.float32, device=device)
        train_anemic_tensor = torch.tensor(train_anemic, dtype=torch.long, device=device)
        train_severity_tensor = torch.tensor(train_severity, dtype=torch.long, device=device)
        train_has_severity_tensor = torch.tensor(train_has_severity, dtype=torch.bool, device=device)

        rng = np.random.default_rng(seed + fold)
        model.train()
        for _ in range(epochs):
            for batch_indices in _iterate_batches(len(train_manifest), batch_size, rng):
                batch_indices_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)
                optimizer.zero_grad()
                outputs = model(train_features_tensor[batch_indices_tensor])
                target_hb = train_hb_tensor[batch_indices_tensor]
                bin_index = model.hb_to_bin_index(target_hb)
                regression_loss = dual_loss_fn(
                    outputs["regression_logits"], outputs["expected_hb"], target_hb, bin_index
                )
                classification_loss = classification_loss_fn(
                    outputs["classification_logits"], train_anemic_tensor[batch_indices_tensor]
                )
                severity_mask = train_has_severity_tensor[batch_indices_tensor]
                if severity_mask.any():
                    severity_loss = severity_loss_fn(
                        outputs["severity_logits"][severity_mask],
                        train_severity_tensor[batch_indices_tensor][severity_mask],
                    )
                else:
                    severity_loss = outputs["expected_hb"].new_zeros(())
                total_loss = (
                    weight_regression * regression_loss
                    + weight_classification * classification_loss
                    + weight_severity * severity_loss
                )
                total_loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_features_tensor = torch.tensor(val_features, dtype=torch.float32, device=device)
            val_outputs = model(val_features_tensor)
            predicted_hb = val_outputs["expected_hb"].cpu().numpy()
            predicted_anemic = val_outputs["classification_logits"].argmax(dim=1).cpu().numpy()
            predicted_severity = CornLoss.predict_rank(val_outputs["severity_logits"]).cpu().numpy()

        mae = float(np.mean(np.abs(predicted_hb - val_hb)))
        rmse = float(np.sqrt(np.mean((predicted_hb - val_hb) ** 2)))
        accuracy = float(np.mean(predicted_anemic == val_anemic))
        if val_has_severity.any():
            severity_accuracy = float(
                np.mean(predicted_severity[val_has_severity] == val_severity[val_has_severity])
            )
        else:
            severity_accuracy = float("nan")

        fold_metrics.append({
            "fold": int(fold),
            "n_val": len(val_manifest),
            "mae": mae,
            "rmse": rmse,
            "accuracy": accuracy,
            "severity_accuracy": severity_accuracy,
        })

        for row_position, uid in enumerate(val_manifest["uid"]):
            oof_records.append({
                "uid": uid,
                "fold": int(fold),
                "hb_true": float(val_hb[row_position]),
                "hb_pred": float(predicted_hb[row_position]),
                "anemic_true": int(val_anemic[row_position]),
                "anemic_pred": int(predicted_anemic[row_position]),
                "severity_true": int(val_severity[row_position]),
                "severity_pred": int(predicted_severity[row_position]),
            })

        models.append(model)

    return {
        "oof": pd.DataFrame(oof_records),
        "fold_metrics": pd.DataFrame(fold_metrics),
        "models": models,
    }
