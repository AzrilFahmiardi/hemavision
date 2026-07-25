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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import features
from src.common.features import EMBEDDING_DIM, build_fusion_input, compute_fusion_stats, freeze_backbone_body
from src.common.models import (
    CornLoss, DualLoss, EndToEndModel, FocalLoss, HemavisionModel, SEVERITY_CLASSES, SEVERITY_ORDER,
)


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


def evaluate_by_dataset(oof: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Hitung MAE, akurasi, dan akurasi severity per dataset dari prediksi out-of-fold.

    Berguna sebagai diagnosis standar untuk mendeteksi apakah performa gabungan
    yang terlihat baik sebenarnya menyembunyikan satu dataset yang collapse ke
    kelas mayoritas, serta sebagai sinyal untuk objective pencarian hyperparameter
    yang sadar keadilan antar dataset.
    """
    merged = oof.merge(manifest[["uid", "dataset"]], on="uid", how="left")
    rows = []
    for dataset_name, group in merged.groupby("dataset"):
        mae = float(np.mean(np.abs(group["hb_pred"] - group["hb_true"])))
        accuracy = float(np.mean(group["anemic_pred"] == group["anemic_true"]))
        severity_valid = group["severity_true"] >= 0
        if severity_valid.any():
            severity_accuracy = float(
                np.mean(group.loc[severity_valid, "severity_pred"] == group.loc[severity_valid, "severity_true"])
            )
        else:
            severity_accuracy = float("nan")
        rows.append({
            "dataset": dataset_name,
            "n": len(group),
            "mae": mae,
            "accuracy": accuracy,
            "severity_accuracy": severity_accuracy,
        })
    return pd.DataFrame(rows)


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
    trunk_dim: int = 128,
    attention_dim: int = 64,
    dropout: float = 0.3,
    focal_gamma: float = 2.0,
    use_handcrafted: bool = True,
    use_deep: bool = True,
    use_fusion_attention: bool = True,
    use_demographics: bool = True,
    use_site_token: bool = True,
    regression_loss: str = "dual",
    weight_severity_classes: bool = False,
    device: str | None = None,
    seed: int = 42,
) -> dict:
    """Latih dan evaluasi model multi-task memakai validasi silang k-fold.

    Manifest harus sudah memiliki kolom fold dari assign_kfold. Parameter
    use_fusion_attention, use_demographics, use_site_token, dan regression_loss
    (dual atau mse_only) memungkinkan ablation komponen arsitektur satu per
    satu dengan protokol pelatihan yang identik. weight_severity_classes
    mengaktifkan bobot kelas berbanding terbalik dengan frekuensi pada CornLoss,
    berguna ketika kelas severity minoritas (misalnya Moderate) jarang muncul.
    Mengembalikan prediksi out-of-fold untuk seluruh sampel, metrik ringkas per
    fold, dan daftar model terlatih per fold untuk keperluan penyimpanan
    checkpoint.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weight_regression, weight_classification, weight_severity = loss_weights
    if regression_loss not in ("dual", "mse_only"):
        raise ValueError(f"regression_loss tidak dikenal: {regression_loss}")

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
            use_demographics=use_demographics, use_site_token=use_site_token,
        )
        val_features = build_fusion_input(
            handcrafted, val_deep, val_manifest, stats=stats,
            use_handcrafted=use_handcrafted, use_deep=use_deep,
            use_demographics=use_demographics, use_site_token=use_site_token,
        )

        train_hb = train_manifest["hb_gdl"].to_numpy(dtype=np.float32)
        val_hb = val_manifest["hb_gdl"].to_numpy(dtype=np.float32)
        train_anemic = train_manifest["anemic"].to_numpy(dtype=np.int64)
        val_anemic = val_manifest["anemic"].to_numpy(dtype=np.int64)
        train_severity = severity_ordinal[train_mask.to_numpy()]
        val_severity = severity_ordinal[val_mask.to_numpy()]
        train_has_severity = has_severity[train_mask.to_numpy()]
        val_has_severity = has_severity[val_mask.to_numpy()]

        model = HemavisionModel(
            input_dim=train_features.shape[1],
            attention_dim=attention_dim,
            trunk_dim=trunk_dim,
            dropout=dropout,
            use_fusion_attention=use_fusion_attention,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        if regression_loss == "dual":
            dual_loss_fn = DualLoss(gamma=focal_gamma)
        else:
            mse_only_fn = torch.nn.MSELoss()
            dual_loss_fn = lambda regression_logits, expected_hb, target_hb, bin_index: mse_only_fn(
                expected_hb, target_hb
            )
        classification_alpha = _class_weights(train_anemic, num_classes=2)
        classification_loss_fn = FocalLoss(gamma=focal_gamma, alpha=classification_alpha)
        if weight_severity_classes and train_has_severity.any():
            severity_class_weights = _class_weights(
                train_severity[train_has_severity], num_classes=SEVERITY_CLASSES
            )
        else:
            severity_class_weights = None
        severity_loss_fn = CornLoss(class_weights=severity_class_weights)

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
            predicted_anemic_prob = torch.softmax(val_outputs["classification_logits"], dim=1)[:, 1].cpu().numpy()
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
                "anemic_prob": float(predicted_anemic_prob[row_position]),
                "severity_true": int(val_severity[row_position]),
                "severity_pred": int(predicted_severity[row_position]),
            })

        models.append(model)

    return {
        "oof": pd.DataFrame(oof_records),
        "fold_metrics": pd.DataFrame(fold_metrics),
        "models": models,
    }


def run_cross_dataset(
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    handcrafted: pd.DataFrame,
    deep_embeddings: np.ndarray | None,
    embedding_uids: list | None,
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 0.5),
    trunk_dim: int = 128,
    attention_dim: int = 64,
    dropout: float = 0.3,
    focal_gamma: float = 2.0,
    use_handcrafted: bool = True,
    use_deep: bool = True,
    device: str | None = None,
    seed: int = 42,
) -> dict:
    """Latih sekali pada satu populasi dan uji pada populasi lain untuk mengukur generalisasi.

    Berbeda dari run_kfold yang melakukan validasi silang di dalam gabungan
    dataset, fungsi ini mengukur seberapa baik model beradaptasi ke populasi
    yang sama sekali tidak terlihat saat pelatihan. Standardisasi fitur
    hand-crafted dihitung dari train_manifest saja. Site token pada skenario ini
    menjadi konstan karena train_manifest berasal dari satu populasi, sehingga
    tidak benar-benar berperan sebagai sinyal adaptasi, dan hal ini dilaporkan
    apa adanya.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weight_regression, weight_classification, weight_severity = loss_weights

    severity_ordinal_train = _severity_to_ordinal(train_manifest["severity"])
    severity_ordinal_test = _severity_to_ordinal(test_manifest["severity"])
    train_has_severity = severity_ordinal_train >= 0
    test_has_severity = severity_ordinal_test >= 0
    site_categories = sorted(pd.concat([train_manifest["site"], test_manifest["site"]]).unique().tolist())

    stats = compute_fusion_stats(handcrafted, train_manifest, site_categories=site_categories)
    train_deep = _select_embeddings(embedding_uids, deep_embeddings, train_manifest["uid"]) if use_deep else None
    test_deep = _select_embeddings(embedding_uids, deep_embeddings, test_manifest["uid"]) if use_deep else None

    train_features = build_fusion_input(
        handcrafted, train_deep, train_manifest, stats=stats, use_handcrafted=use_handcrafted, use_deep=use_deep,
    )
    test_features = build_fusion_input(
        handcrafted, test_deep, test_manifest, stats=stats, use_handcrafted=use_handcrafted, use_deep=use_deep,
    )

    train_hb = train_manifest["hb_gdl"].to_numpy(dtype=np.float32)
    test_hb = test_manifest["hb_gdl"].to_numpy(dtype=np.float32)
    train_anemic = train_manifest["anemic"].to_numpy(dtype=np.int64)
    test_anemic = test_manifest["anemic"].to_numpy(dtype=np.int64)

    model = HemavisionModel(
        input_dim=train_features.shape[1], attention_dim=attention_dim, trunk_dim=trunk_dim, dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    dual_loss_fn = DualLoss(gamma=focal_gamma)
    classification_alpha = _class_weights(train_anemic, num_classes=2)
    classification_loss_fn = FocalLoss(gamma=focal_gamma, alpha=classification_alpha)
    severity_loss_fn = CornLoss()

    train_features_tensor = torch.tensor(train_features, dtype=torch.float32, device=device)
    train_hb_tensor = torch.tensor(train_hb, dtype=torch.float32, device=device)
    train_anemic_tensor = torch.tensor(train_anemic, dtype=torch.long, device=device)
    train_severity_tensor = torch.tensor(severity_ordinal_train, dtype=torch.long, device=device)
    train_has_severity_tensor = torch.tensor(train_has_severity, dtype=torch.bool, device=device)

    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        for batch_indices in _iterate_batches(len(train_manifest), batch_size, rng):
            batch_indices_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)
            optimizer.zero_grad()
            outputs = model(train_features_tensor[batch_indices_tensor])
            target_hb = train_hb_tensor[batch_indices_tensor]
            bin_index = model.hb_to_bin_index(target_hb)
            regression_loss = dual_loss_fn(outputs["regression_logits"], outputs["expected_hb"], target_hb, bin_index)
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
        test_features_tensor = torch.tensor(test_features, dtype=torch.float32, device=device)
        test_outputs = model(test_features_tensor)
        predicted_hb = test_outputs["expected_hb"].cpu().numpy()
        predicted_anemic = test_outputs["classification_logits"].argmax(dim=1).cpu().numpy()

    mae = float(np.mean(np.abs(predicted_hb - test_hb)))
    accuracy = float(np.mean(predicted_anemic == test_anemic))
    true_positive = int(np.sum((predicted_anemic == 1) & (test_anemic == 1)))
    true_negative = int(np.sum((predicted_anemic == 0) & (test_anemic == 0)))
    sensitivity = true_positive / int(np.sum(test_anemic == 1)) if int(np.sum(test_anemic == 1)) else float("nan")
    specificity = true_negative / int(np.sum(test_anemic == 0)) if int(np.sum(test_anemic == 0)) else float("nan")
    balanced_accuracy = (sensitivity + specificity) / 2.0

    return {
        "mae": mae,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "model": model,
    }


def recompute_oof_probabilities(
    manifest: pd.DataFrame,
    handcrafted: pd.DataFrame,
    deep_embeddings: np.ndarray | None,
    embedding_uids: list | None,
    checkpoint_paths: list,
    trunk_dim: int = 128,
    attention_dim: int = 64,
    dropout: float = 0.3,
    use_handcrafted: bool = True,
    use_deep: bool = True,
    use_demographics: bool = True,
    use_site_token: bool = True,
    device: str | None = None,
) -> pd.DataFrame:
    """Hitung ulang probabilitas klasifikasi anemia dari checkpoint per fold yang sudah dilatih.

    Dipakai ketika prediksi out-of-fold yang tersimpan hanya memuat kelas hasil
    argmax, bukan probabilitas kontinu yang dibutuhkan untuk analisis ROC dan
    threshold Youden. Fungsi ini murni melakukan inferensi ulang tanpa melatih
    ulang model, sehingga hasilnya identik dengan model yang sudah tersimpan.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    site_categories = sorted(manifest["site"].unique().tolist())
    records = []
    for fold, checkpoint_path in zip(sorted(manifest["fold"].unique()), checkpoint_paths):
        train_mask = manifest["fold"] != fold
        val_mask = manifest["fold"] == fold
        train_manifest = manifest[train_mask].reset_index(drop=True)
        val_manifest = manifest[val_mask].reset_index(drop=True)

        stats = compute_fusion_stats(handcrafted, train_manifest, site_categories=site_categories)
        val_deep = _select_embeddings(embedding_uids, deep_embeddings, val_manifest["uid"]) if use_deep else None
        val_features = build_fusion_input(
            handcrafted, val_deep, val_manifest, stats=stats,
            use_handcrafted=use_handcrafted, use_deep=use_deep,
            use_demographics=use_demographics, use_site_token=use_site_token,
        )

        model = HemavisionModel(
            input_dim=val_features.shape[1], attention_dim=attention_dim, trunk_dim=trunk_dim, dropout=dropout,
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            val_features_tensor = torch.tensor(val_features, dtype=torch.float32, device=device)
            outputs = model(val_features_tensor)
            probability = torch.softmax(outputs["classification_logits"], dim=1)[:, 1].cpu().numpy()

        for row_position, uid in enumerate(val_manifest["uid"]):
            records.append({"uid": uid, "fold": int(fold), "anemic_prob": float(probability[row_position])})

    return pd.DataFrame(records)


class EndToEndDataset(Dataset):
    """Dataset citra beserta fitur statis dan target untuk pelatihan end-to-end.

    Fitur statis (hand-crafted terstandardisasi, demografi, site token) sudah
    dihitung sebelumnya karena deterministik dan tidak memerlukan gradien,
    sedangkan citra dimuat per sampel untuk diproses lewat backbone yang
    dilatih bersama.
    """

    def __init__(self, frame: pd.DataFrame, static_features: np.ndarray, severity_ordinal: np.ndarray,
                 has_severity: np.ndarray, size: int = 224, augment: bool = False):
        self.rows = frame.reset_index(drop=True)
        self.static_features = static_features
        self.severity_ordinal = severity_ordinal
        self.has_severity = has_severity
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image = features.load_roi_tensor(row["roi_path"], size=self.size, augment=self.augment)
        static = torch.tensor(self.static_features[index], dtype=torch.float32)
        hb = torch.tensor(row["hb_gdl"], dtype=torch.float32)
        anemic = torch.tensor(row["anemic"], dtype=torch.long)
        severity = torch.tensor(self.severity_ordinal[index], dtype=torch.long)
        has_sev = torch.tensor(bool(self.has_severity[index]), dtype=torch.bool)
        return image, static, hb, anemic, severity, has_sev


def run_kfold_end_to_end(
    manifest: pd.DataFrame,
    handcrafted: pd.DataFrame,
    n_splits: int = 5,
    epochs: int = 60,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 0.5),
    trunk_dim: int = 128,
    attention_dim: int = 64,
    dropout: float = 0.3,
    focal_gamma: float = 2.0,
    image_size: int = 224,
    backbone_name: str = "resnet18",
    augment: bool = False,
    weight_severity_classes: bool = False,
    device: str | None = None,
    seed: int = 42,
) -> dict:
    """Latih model multi-task end-to-end, backbone dibekukan kecuali CSA dan proyeksi.

    Berbeda dari run_kfold yang memakai deep embedding yang sudah dibekukan,
    fungsi ini memuat citra langsung dan melatih CSA beserta layer proyeksi
    lewat backpropagation penuh, sehingga atensi yang dihasilkan benar-benar
    dipelajari dari data, bukan inisialisasi acak. Badan backbone (early dan
    late) tetap dibekukan mengikuti freeze_backbone_body. augment mengaktifkan
    augmentasi citra ringan (flip, rotasi kecil, brightness/contrast) pada
    split train saja, dipakai untuk mengurangi overfitting dibanding percobaan
    fine-tuning end-to-end sebelumnya yang tidak memakai augmentasi sama sekali.
    weight_severity_classes mengaktifkan bobot kelas pada CornLoss, sama seperti
    pada run_kfold.
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
        train_static = build_fusion_input(
            handcrafted, None, train_manifest, stats=stats, use_handcrafted=True, use_deep=False,
        )
        val_static = build_fusion_input(
            handcrafted, None, val_manifest, stats=stats, use_handcrafted=True, use_deep=False,
        )

        train_severity = severity_ordinal[train_mask.to_numpy()]
        val_severity = severity_ordinal[val_mask.to_numpy()]
        train_has_severity = has_severity[train_mask.to_numpy()]
        val_has_severity = has_severity[val_mask.to_numpy()]

        train_dataset = EndToEndDataset(
            train_manifest, train_static, train_severity, train_has_severity, size=image_size, augment=augment,
        )
        val_dataset = EndToEndDataset(
            val_manifest, val_static, val_severity, val_has_severity, size=image_size, augment=False,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        embedding_backbone = features.EmbeddingBackbone(backbone_name=backbone_name)
        freeze_backbone_body(embedding_backbone)
        model = EndToEndModel(
            embedding_backbone, static_dim=train_static.shape[1], embedding_dim=EMBEDDING_DIM,
            attention_dim=attention_dim, trunk_dim=trunk_dim, dropout=dropout,
        ).to(device)

        trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        dual_loss_fn = DualLoss(gamma=focal_gamma)
        classification_alpha = _class_weights(train_manifest["anemic"].to_numpy(dtype=np.int64), num_classes=2)
        classification_loss_fn = FocalLoss(gamma=focal_gamma, alpha=classification_alpha)
        if weight_severity_classes and train_has_severity.any():
            severity_class_weights = _class_weights(
                train_severity[train_has_severity], num_classes=SEVERITY_CLASSES
            )
        else:
            severity_class_weights = None
        severity_loss_fn = CornLoss(class_weights=severity_class_weights)

        model.train()
        for _ in range(epochs):
            for images, static, hb, anemic, severity, has_sev in train_loader:
                images, static = images.to(device), static.to(device)
                hb, anemic = hb.to(device), anemic.to(device)
                severity, has_sev = severity.to(device), has_sev.to(device)

                optimizer.zero_grad()
                outputs = model(images, static)
                bin_index = model.hb_to_bin_index(hb)
                regression_loss = dual_loss_fn(outputs["regression_logits"], outputs["expected_hb"], hb, bin_index)
                classification_loss = classification_loss_fn(outputs["classification_logits"], anemic)
                if has_sev.any():
                    severity_loss = severity_loss_fn(outputs["severity_logits"][has_sev], severity[has_sev])
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
        n_val = len(val_manifest)
        predicted_hb = np.zeros(n_val, dtype=np.float32)
        predicted_anemic = np.zeros(n_val, dtype=np.int64)
        predicted_severity = np.zeros(n_val, dtype=np.int64)
        position = 0
        with torch.no_grad():
            for images, static, hb, anemic, severity, has_sev in val_loader:
                images, static = images.to(device), static.to(device)
                outputs = model(images, static)
                batch_n = images.shape[0]
                predicted_hb[position:position + batch_n] = outputs["expected_hb"].cpu().numpy()
                predicted_anemic[position:position + batch_n] = (
                    outputs["classification_logits"].argmax(dim=1).cpu().numpy()
                )
                predicted_severity[position:position + batch_n] = (
                    CornLoss.predict_rank(outputs["severity_logits"]).cpu().numpy()
                )
                position += batch_n

        val_hb = val_manifest["hb_gdl"].to_numpy(dtype=np.float32)
        val_anemic = val_manifest["anemic"].to_numpy(dtype=np.int64)

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
            "n_val": n_val,
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
