"""
Modul Particle Swarm Optimization (PSO) untuk Tuning Hyperparameter LSTM
Kelompok 6 - Household Electricity Load Forecasting
Mata Kuliah: Kecerdasan Buatan (TE2514031) - Institut Teknologi Kalimantan
"""

import numpy as np
import time


class PSO_LSTM_Optimizer:
    """
    Particle Swarm Optimization (PSO) untuk mengoptimasi:
    - sequence_length: rentang [12, 36] jam
    - lstm_units: rentang [16, 64] unit
    - dropout_rate: rentang [0.05, 0.35]
    - learning_rate: rentang [0.0003, 0.005] (skala log10 [-3.5, -2.3])
    - batch_size: pilihan [32, 64]
    """

    def __init__(self, n_particles: int = 8,
                 max_iter: int = 6,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 random_seed: int = 42):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.RandomState(random_seed)

        # Batas pencarian (Search Bounds): [min, max]
        # Dimensi 0: seq_len (12..36)
        # Dimensi 1: lstm_units (16..64)
        # Dimensi 2: dropout (0.05..0.35)
        # Dimensi 3: log10_lr (-3.5 .. -2.3)
        # Dimensi 4: batch_size_idx (0..1 -> 32 atau 64)
        self.bounds_min = np.array([12.0, 16.0, 0.05, -3.5, 0.0])
        self.bounds_max = np.array([36.0, 64.0, 0.35, -2.3, 1.99])
        self.dim = len(self.bounds_min)

        self.gbest_position = None
        self.gbest_score = np.inf
        self.history_gbest = []

    def initialize_swarm(self):
        """Inisialisasi posisi dan kecepatan partikel."""
        positions = self.rng.uniform(
            self.bounds_min, self.bounds_max,
            size=(self.n_particles, self.dim)
        )
        velocities = self.rng.uniform(
            -0.1 * (self.bounds_max - self.bounds_min),
            0.1 * (self.bounds_max - self.bounds_min),
            size=(self.n_particles, self.dim)
        )
        return positions, velocities

    def decode_particle(self, position: np.ndarray) -> dict:
        """Mengonversi vektor kontinu partikel menjadi parameter fisik LSTM."""
        clipped = np.clip(position, self.bounds_min, self.bounds_max)
        seq_len = int(np.round(clipped[0]))
        lstm_units = int(np.round(clipped[1] / 8.0) * 8)  # kelipatan 8
        lstm_units = max(16, min(64, lstm_units))
        dropout = float(round(clipped[2], 3))
        lr = float(10 ** clipped[3])
        batch_size = 32 if clipped[4] < 1.0 else 64

        return {
            'sequence_length': seq_len,
            'lstm_units': lstm_units,
            'dropout_rate': dropout,
            'learning_rate': lr,
            'batch_size': batch_size
        }
