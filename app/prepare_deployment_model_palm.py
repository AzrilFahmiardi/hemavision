"""Siapkan model palm yang benar-benar dapat dipakai untuk citra baru.

Bug yang sama dengan konjungtiva berlaku di sini: deep embedding Stage 4
palm dihasilkan dari EmbeddingBackbone dengan CSA dan layer proyeksi
berinisialisasi acak yang tidak pernah disimpan sebagai checkpoint, hanya
embedding hasilnya yang disimpan (lihat
docs/MODEL_ARCHITECTURE_CONJUNCTIVA.md, bagian "Perbaikan Reproducibility
Embedding", untuk penjelasan akar masalah). Akibatnya, backbone segar yang
dibuat ulang saat inference memakai proyeksi acak yang berbeda sama sekali
dari yang dipelajari HemavisionModel selama pelatihan.

Skrip ini memperbaiki akar masalah tersebut untuk palm: ekstraksi embedding
diulang dengan seed tetap, backbone (termasuk CSA dan proyeksi) disimpan
sebagai checkpoint, lalu model produksi (Full Fusion, ResNet18-CSA, bobot
kelas severity aktif pada CornLoss) dilatih ulang lima-fold pada embedding
baru ini memakai hyperparameter default run_kfold yang sama seperti notebook
05_multitask_train.ipynb sel weighted_severity_result (tidak ada pencarian
Optuna terpisah untuk konfigurasi ini, berbeda dari konjungtiva).

Jalankan sekali setiap kali model final untuk deployment berganti:
    python -m app.prepare_deployment_model_palm
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import artifacts_dir, outputs_dir
from src.common import features, manifest as manifest_utils, train

EMBEDDING_SEED = 42


def main() -> None:
    output_dir = outputs_dir("palm")
    artifact_dir = artifacts_dir("palm")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(output_dir / "manifest.csv")
    manifest = manifest[manifest["roi_precropped"]].reset_index(drop=True)
    manifest = manifest_utils.assign_kfold(manifest, n_splits=5, seed=42)
    handcrafted = pd.read_csv(output_dir / "handcrafted_features.csv")

    print("mengekstrak ulang embedding ResNet18-CSA dengan seed tetap...")
    backbone = features.EmbeddingBackbone(backbone_name="resnet18", seed=EMBEDDING_SEED)
    embeddings, embedding_uids = features.extract_deep_embeddings(manifest, model=backbone)

    torch.save(backbone.state_dict(), artifact_dir / "embedding_backbone_resnet18.pt")
    np.save(output_dir / "deep_embeddings.npy", embeddings)
    pd.DataFrame({"uid": embedding_uids}).to_csv(output_dir / "deep_embeddings_uids.csv", index=False)
    print("backbone dan embedding baru disimpan")

    print("melatih ulang Full Fusion ResNet18-CSA lima-fold dengan bobot kelas severity...")
    result = train.run_kfold(
        manifest, handcrafted, embeddings, embedding_uids,
        n_splits=5, epochs=60, use_handcrafted=True, use_deep=True, weight_severity_classes=True,
    )

    print(result["fold_metrics"].round(4).to_string(index=False))
    print("mae rata-rata", round(result["fold_metrics"]["mae"].mean(), 4))
    print("akurasi rata-rata", round(result["fold_metrics"]["accuracy"].mean(), 4))

    result["oof"].to_csv(output_dir / "multitask_oof_full_fusion_weighted_severity_deployment.csv", index=False)
    result["fold_metrics"].to_csv(output_dir / "multitask_fold_metrics_weighted_severity_deployment.csv", index=False)
    for fold_index, fold_model in enumerate(result["models"]):
        torch.save(fold_model.state_dict(), artifact_dir / f"multitask_full_fusion_resnet18_fold{fold_index}.pt")
    print("checkpoint model deployment-ready disimpan ke", artifact_dir)


if __name__ == "__main__":
    main()
