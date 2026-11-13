"""
Modul Genetic Algorithm (GA) untuk Seleksi Fitur dan Lag Beban Listrik
Kelompok 6 - Household Electricity Load Forecasting
Mata Kuliah: Kecerdasan Buatan (TE2514031) - Institut Teknologi Kalimantan
"""

import numpy as np
import time


class GeneticFeatureSelector:
    """
    Genetic Algorithm untuk seleksi subset variabel listrik dan lag waktu.
    Mengoptimalkan kompromi antara akurasi peramalan (rendahnya RMSE)
    dan kekompakan representasi (sedikitnya jumlah fitur).
    """

    def __init__(self, feature_names: list,
                 population_size: int = 12,
                 generations: int = 10,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = None,
                 tournament_size: int = 3,
                 elitism_count: int = 2,
                 complexity_penalty: float = 0.05,
                 random_seed: int = 42):
        self.feature_names = feature_names
        self.n_genes = len(feature_names)
        self.pop_size = population_size
        self.generations = generations
        self.pc = crossover_rate
        self.pm = mutation_rate if mutation_rate else (1.0 / self.n_genes)
        self.k = tournament_size
        self.n_elite = elitism_count
        self.alpha = complexity_penalty
        self.seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        self.best_chromosome = None
        self.best_fitness = -np.inf
        self.best_rmse = np.inf
        self.history_best_fitness = []
        self.history_mean_fitness = []
        self.history_best_features = []

    def initialize_population(self) -> np.ndarray:
        """Inisialisasi populasi acak (vektor biner). Setiap individu memiliki minimal 2 fitur aktif."""
        pop = self.rng.randint(0, 2, size=(self.pop_size, self.n_genes))
        for i in range(self.pop_size):
            if np.sum(pop[i]) == 0:
                idx = self.rng.choice(self.n_genes, size=3, replace=False)
                pop[i, idx] = 1
        return pop

    def tournament_selection(self, population: np.ndarray, fitness_scores: np.ndarray) -> np.ndarray:
        """Tournament selection tanpa bias."""
        selected = []
        for _ in range(self.pop_size):
            aspirants = self.rng.choice(self.pop_size, size=self.k, replace=False)
            best_idx = aspirants[np.argmax(fitness_scores[aspirants])]
            selected.append(population[best_idx].copy())
        return np.array(selected)

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        """Uniform crossover."""
        if self.rng.rand() < self.pc:
            mask = self.rng.randint(0, 2, size=self.n_genes).astype(bool)
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            # Pastikan minimal 1 gen aktif
            if np.sum(child1) == 0:
                child1[self.rng.randint(0, self.n_genes)] = 1
            if np.sum(child2) == 0:
                child2[self.rng.randint(0, self.n_genes)] = 1
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Bit-flip mutation dengan probabilitas Pm."""
        mutated = chromosome.copy()
        for g in range(self.n_genes):
            if self.rng.rand() < self.pm:
                mutated[g] = 1 - mutated[g]
        if np.sum(mutated) == 0:
            mutated[self.rng.randint(0, self.n_genes)] = 1
        return mutated
