"""
Modul Evaluasi dan Metrik Peramalan Runtun Waktu Listrik
Kelompok 6 - Household Electricity Load Forecasting
Mata Kuliah: Kecerdasan Buatan (TE2514031) - Institut Teknologi Kalimantan
"""

import numpy as np


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Menghitung metrik performa regresi peramalan beban listrik:
    1. MAE  (Mean Absolute Error) [kW]
    2. RMSE (Root Mean Squared Error) [kW]
    3. R2   (Coefficient of Determination) [-]
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Koefisien Determinasi R2
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    if ss_total == 0.0:
        r2 = 0.0
    else:
        r2 = float(1.0 - (ss_residual / ss_total))

    # MAPE (Mean Absolute Percentage Error) dengan proteksi pembagian nol
    non_zero_mask = y_true > 0.05
    if np.sum(non_zero_mask) > 0:
        mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0)
    else:
        mape = 0.0

    return {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4),
        'MAPE_pct': round(mape, 2)
    }


def denormalize_predictions(y_scaled: np.ndarray, target_scaler) -> np.ndarray:
    """
    Mengembalikan nilai prediksi yang ternormalisasi [0, 1]
    kembali ke skala fisik aslinya (kW).
    """
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    return target_scaler.inverse_transform(y_scaled).flatten()
