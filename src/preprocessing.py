"""
Modul Preprocessing dan Feature Engineering Runtun Waktu Listrik
Kelompok 6 - Household Electricity Load Forecasting
Mata Kuliah: Kecerdasan Buatan (TE2514031) - Institut Teknologi Kalimantan
"""

import os
import pickle
import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    class MinMaxScaler:
        """Fallback Pure-Numpy MinMaxScaler jika scikit-learn belum terinstal."""
        def __init__(self, feature_range=(0, 1)):
            self.feature_range = feature_range
            self.data_min_ = None
            self.data_max_ = None
            self.scale_ = None
            self.min_ = None

        def fit(self, X):
            X = np.asarray(X, dtype=np.float64)
            self.data_min_ = np.nanmin(X, axis=0)
            self.data_max_ = np.nanmax(X, axis=0)
            diff = self.data_max_ - self.data_min_
            diff = np.where(diff == 0.0, 1.0, diff)
            self.scale_ = (self.feature_range[1] - self.feature_range[0]) / diff
            self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
            return self

        def transform(self, X):
            X = np.asarray(X, dtype=np.float64)
            return X * self.scale_ + self.min_

        def fit_transform(self, X):
            return self.fit(X).transform(X)

        def inverse_transform(self, X):
            X = np.asarray(X, dtype=np.float64)
            return (X - self.min_) / self.scale_


def load_raw_uci_data(filepath: str, nrows: int = None) -> pd.DataFrame:
    """
    Membaca dataset mentah UCI 'household_power_consumption.txt'.
    Menangani missing value berupa tanda '?' menjadi NaN,
    dan memparsing kolom Date (DD/MM/YYYY) dan Time (HH:MM:SS) menjadi DatetimeIndex.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {filepath}")

    print(f"[PREPROCESS] Membaca raw dataset dari: {filepath} ...")
    df = pd.read_csv(
        filepath,
        sep=';',
        na_values='?',
        low_memory=False,
        nrows=nrows
    )

    # Gabungkan Date dan Time menjadi satu DateTimeIndex
    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S'
    )
    df = df.set_index('Datetime').drop(columns=['Date', 'Time'])

    # Pastikan seluruh kolom bertipe numerik float64
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"[PREPROCESS] Dataset terbaca: {df.shape[0]} baris, {df.shape[1]} kolom.")
    return df


def clean_and_aggregate_hourly(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan missing value interval pendek dengan interpolasi linier,
    melakukan agregasi per jam (1-Hour Mean) untuk memodelkan daya aktif rata-rata,
    dan menghitung fitur Unmetered Active Energy (Sub_metering_remainder).
    """
    print("[PREPROCESS] Menangani missing values dan melakukan agregasi per jam (1h)...")

    # Interpolasi linier pada data menit (maksimal jeda 60 menit)
    df_clean = df_raw.interpolate(method='linear', limit=60)
    df_clean = df_clean.bfill().ffill()

    # Agregasi rata-rata per jam (Mean Aggregation)
    df_hourly = df_clean.resample('1h').mean()

    # Tangani sisa gap jam (jika ada pemadaman berhari-hari) secara linier
    df_hourly = df_hourly.interpolate(method='linear').bfill().ffill()

    # Feature Engineering Domain Elektro: Beban Tak Terukur (Sub_metering_remainder)
    # Total active energy Wh per menit = GAP * 1000 / 60
    # Pada data per jam, rata-rata Wh per menit = (GAP * 1000 / 60) - (Sub1 + Sub2 + Sub3)
    df_hourly['Sub_metering_remainder'] = (
        (df_hourly['Global_active_power'] * 1000.0 / 60.0) -
        (df_hourly['Sub_metering_1'] + df_hourly['Sub_metering_2'] + df_hourly['Sub_metering_3'])
    )
    # Sensor jitter clipping (mencegah nilai negatif akibat toleransi presisi alat ukur)
    df_hourly['Sub_metering_remainder'] = df_hourly['Sub_metering_remainder'].clip(lower=0.0)

    print(f"[PREPROCESS] Agregasi selesai: {df_hourly.shape[0]} titik data per jam.")
    return df_hourly


def generate_lag_and_temporal_features(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Membangun lag features untuk Global_active_power (t-1, t-2, t-3, t-24),
    rolling mean 24 jam, serta fitur siklis waktu (hour sin/cos, day sin/cos, weekend).
    """
    print("[PREPROCESS] Membangun lag features dan fitur temporal siklis...")
    df_feat = df_hourly.copy()

    # Lag features target (beban listrik masa lalu)
    df_feat['gap_lag_1'] = df_feat['Global_active_power'].shift(1)
    df_feat['gap_lag_2'] = df_feat['Global_active_power'].shift(2)
    df_feat['gap_lag_3'] = df_feat['Global_active_power'].shift(3)
    df_feat['gap_lag_24'] = df_feat['Global_active_power'].shift(24)

    # Rolling mean 24 jam (menggambarkan level konsumsi harian tanpa bocor ke masa depan)
    df_feat['rolling_mean_24h'] = df_feat['Global_active_power'].shift(1).rolling(window=24).mean()

    # Fitur siklis waktu (Representasi trigonometri siklus waktu sirkadian dan mingguan)
    hours = df_feat.index.hour
    dayofweek = df_feat.index.dayofweek

    df_feat['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df_feat['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
    df_feat['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / 7.0)
    df_feat['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / 7.0)
    df_feat['is_weekend'] = (dayofweek >= 5).astype(int)

    # Buang baris awal NaN akibat lag 24
    df_clean = df_feat.dropna().copy()
    print(f"[PREPROCESS] Total baris bersih setelah pembentukan fitur: {df_clean.shape[0]} baris, {df_clean.shape[1]} kolom.")
    return df_clean


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """
    Membagi dataset secara kronologis murni (strictly time-ordered split).
    Mencegah data leakage.
    Returns: train_df, val_df, test_df
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"[SPLIT] Chronological Split Terbentuk:")
    print(f"  - Train Set      : {len(train_df)} baris ({train_df.index.min()} s.d. {train_df.index.max()})")
    print(f"  - Validation Set : {len(val_df)} baris ({val_df.index.min()} s.d. {val_df.index.max()})")
    print(f"  - Test Set       : {len(test_df)} baris ({test_df.index.min()} s.d. {test_df.index.max()})")

    return train_df, val_df, test_df


def scale_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                   feature_cols: list, target_col: str = 'Global_active_power'):
    """
    Melakukan normalisasi MinMaxScaler pada rentang [0, 1].
    PENTING: Scaler di-fit HANYA pada train_df untuk mencegah kebocoran data.
    """
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit hanya pada training
    feature_scaler.fit(train_df[feature_cols])
    target_scaler.fit(train_df[[target_col]])

    # Transformasikan tiap subset
    X_train_scaled = feature_scaler.transform(train_df[feature_cols])
    y_train_scaled = target_scaler.transform(train_df[[target_col]]).flatten()

    X_val_scaled = feature_scaler.transform(val_df[feature_cols])
    y_val_scaled = target_scaler.transform(val_df[[target_col]]).flatten()

    X_test_scaled = feature_scaler.transform(test_df[feature_cols])
    y_test_scaled = target_scaler.transform(test_df[[target_col]]).flatten()

    return {
        'X_train': X_train_scaled, 'y_train': y_train_scaled,
        'X_val': X_val_scaled,     'y_val': y_val_scaled,
        'X_test': X_test_scaled,   'y_test': y_test_scaled,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }


def create_sliding_windows(X_data: np.ndarray, y_data: np.ndarray, sequence_length: int):
    """
    Membangun 3D sliding window tensor untuk LSTM:
    Bentuk Output:
    - X_seq: (N - sequence_length, sequence_length, n_features)
    - y_seq: (N - sequence_length,) -> target t+1 berikutnya
    """
    X_seq, y_seq = [], []
    for i in range(len(X_data) - sequence_length):
        X_seq.append(X_data[i:i + sequence_length])
        y_seq.append(y_data[i + sequence_length])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)
