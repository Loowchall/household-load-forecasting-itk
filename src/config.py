"""
Konfigurasi Global Proyek Tugas Besar Kecerdasan Buatan (TE2514031)
Kelompok 6 - Household Electricity Load Forecasting
Dataset: UCI ID 235 (Individual Household Electric Power Consumption)
"""

import os
from pathlib import Path

# Random Seed untuk Reproducibility
RANDOM_SEED = 42

# Resolusi Waktu & Target
TIME_RESOLUTION = '1h'  # Agregasi 1 jam (Short-Term Load Forecasting)
TARGET_COLUMN = 'Global_active_power'  # Target: Daya aktif rata-rata per jam (kW)

# Fitur Listrik Asli UCI
RAW_NUMERIC_COLS = [
    'Global_active_power',
    'Global_reactive_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3',
]

# Seluruh Kandidat Fitur untuk Seleksi GA (17 Fitur)
CANDIDATE_FEATURES = [
    'Global_reactive_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3',
    'Sub_metering_remainder',
    'gap_lag_1',
    'gap_lag_2',
    'gap_lag_3',
    'gap_lag_24',
    'rolling_mean_24h',
    'hour_sin',
    'hour_cos',
    'dayofweek_sin',
    'dayofweek_cos',
    'is_weekend',
]

# Rasio Pembagian Data (Chronological Split)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Parameter Default Baseline LSTM (Model A)
BASELINE_SEQUENCE_LENGTH = 24  # Riwayat 24 jam ke belakang
BASELINE_LSTM_UNITS = 32
BASELINE_DROPOUT = 0.1
BASELINE_LEARNING_RATE = 0.001
BASELINE_BATCH_SIZE = 32
BASELINE_EPOCHS = 20

# Search Space untuk PSO (Model C & D)
PSO_BOUNDS = {
    'seq_len': (12, 48),         # Jam
    'lstm_units': (16, 128),     # Jumlah neuron
    'dropout': (0.0, 0.4),       # Dropout rate
    'learning_rate': (1e-4, 1e-2), # Skala log
    'batch_size': [16, 32, 64],  # Diskrit
}

# Linguistic Variables untuk Fuzzy Inference System (5 Kategori Dosen)
FUZZY_STATUS_CLASSES = ['Very Low', 'Low', 'Normal', 'High', 'Peak']

# Path Direktori (Agnostik Lingkungan - Mendukung Lokal & Colab)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
