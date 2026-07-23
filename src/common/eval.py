"""Evaluation utilities.

Menyediakan metrik klinis yang tidak tersedia langsung sebagai satu pemanggilan
di scikit-learn, yaitu analisis Bland Altman untuk regresi hemoglobin, threshold
operasi optimal menurut indeks Youden untuk klasifikasi anemia, dan PPV/NPV dari
confusion matrix. Metrik lain seperti ROC, AUC, F1, confusion matrix, dan Cohen
kappa dipanggil langsung dari scikit-learn di notebook karena sudah tersedia
sebagai satu pemanggilan fungsi.
"""
from __future__ import annotations

import numpy as np


def bland_altman_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Hitung bias dan limits of agreement analisis Bland Altman.

    Bias adalah rerata selisih prediksi terhadap nilai sebenarnya, sedangkan
    limits of agreement adalah rentang bias ditambah dan dikurangi 1.96 kali
    simpangan baku selisih, mengikuti konvensi baku analisis Bland Altman.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    difference = y_pred - y_true
    mean_value = (y_true + y_pred) / 2.0
    bias = float(difference.mean())
    standard_deviation = float(difference.std(ddof=1))
    lower_limit = bias - 1.96 * standard_deviation
    upper_limit = bias + 1.96 * standard_deviation
    return {
        "bias": bias,
        "standard_deviation": standard_deviation,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "mean_value": mean_value,
        "difference": difference,
    }


def youden_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Cari threshold probabilitas yang memaksimalkan indeks Youden.

    Indeks Youden adalah sensitivitas ditambah spesifisitas dikurangi satu.
    Threshold ini dipilih karena skrining anemia mengutamakan sensitivitas
    tinggi, sejalan dengan penekanan Paper 1 dan Paper B pada literatur.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    candidate_thresholds = np.unique(y_prob)
    best_threshold = 0.5
    best_youden = -1.0
    best_sensitivity = 0.0
    best_specificity = 0.0
    for threshold in candidate_thresholds:
        prediction = (y_prob >= threshold).astype(int)
        true_positive = int(np.sum((prediction == 1) & (y_true == 1)))
        false_negative = int(np.sum((prediction == 0) & (y_true == 1)))
        true_negative = int(np.sum((prediction == 0) & (y_true == 0)))
        false_positive = int(np.sum((prediction == 1) & (y_true == 0)))
        sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else 0.0
        youden = sensitivity + specificity - 1.0
        if youden > best_youden:
            best_youden = youden
            best_threshold = float(threshold)
            best_sensitivity = sensitivity
            best_specificity = specificity
    return {
        "threshold": best_threshold,
        "youden_index": best_youden,
        "sensitivity": best_sensitivity,
        "specificity": best_specificity,
    }


def ppv_npv(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Hitung positive predictive value dan negative predictive value.

    Kedua metrik ini melengkapi presisi dan recall standar karena menyatakan
    keandalan hasil positif dan negatif dari sudut pandang klinis.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_positive = int(np.sum((y_pred == 1) & (y_true == 1)))
    false_positive = int(np.sum((y_pred == 1) & (y_true == 0)))
    true_negative = int(np.sum((y_pred == 0) & (y_true == 0)))
    false_negative = int(np.sum((y_pred == 0) & (y_true == 1)))
    ppv = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else float("nan")
    npv = true_negative / (true_negative + false_negative) if (true_negative + false_negative) else float("nan")
    return {"ppv": ppv, "npv": npv}


def regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Ringkasan metrik regresi hemoglobin: MAE, RMSE, dan R squared."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    total_variance = float(np.sum((y_true - y_true.mean()) ** 2))
    residual_variance = float(np.sum(error ** 2))
    r_squared = 1.0 - residual_variance / total_variance if total_variance > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r_squared": r_squared}
