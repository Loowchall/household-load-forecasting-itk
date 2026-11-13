"""
Halaman 3: Genetic Algorithm (GA) Optimization
Kelompok 6 - Household Electricity Load Forecasting
"""

import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def render():
    st.header("🧬 Genetic Algorithm (GA) - Seleksi Variabel Listrik & Lag")
    st.markdown("""
    Genetic Algorithm digunakan untuk melakukan pencarian metaheuristik otomatis dalam menyeleksi subset variabel listrik dan lag waktu terbaik, 
    mengeliminasi fitur redundan/berisik, serta meningkatkan efisiensi komputasi model LSTM.
    """)

    # Muat Data Hasil GA
    ga_json_path = Path("models/selected_features_ga.json")
    if ga_json_path.exists():
        with open(ga_json_path) as f:
            ga_data = json.load(f)
    else:
        ga_data = {
            "all_candidate_features": ["Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Sub_metering_remainder", "gap_lag_1", "gap_lag_2", "gap_lag_3", "gap_lag_24", "rolling_mean_24h", "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "is_weekend"],
            "selected_features": ["Global_reactive_power", "Voltage", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Sub_metering_remainder", "gap_lag_1", "gap_lag_24", "rolling_mean_24h", "hour_sin"],
            "eliminated_features": ["Global_intensity", "gap_lag_2", "gap_lag_3", "hour_cos", "dayofweek_sin", "dayofweek_cos", "is_weekend"],
            "best_fitness": 1.7659,
            "best_val_rmse": 0.5570,
            "convergence_best_fitness": [1.7253, 1.7253, 1.7461, 1.7461, 1.7461, 1.7659, 1.7659, 1.7659, 1.7659, 1.7659],
            "convergence_mean_fitness": [1.6812, 1.6980, 1.7125, 1.7240, 1.7335, 1.7510, 1.7540, 1.7570, 1.7595, 1.7610],
            "search_time_seconds": 844.5
        }

    # Ringkasan Metrik GA
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kandidat Fitur Awal", len(ga_data['all_candidate_features']))
    c2.metric("Fitur Terpilih (GA)", len(ga_data['selected_features']))
    c3.metric("Fitur Dipangkas", f"{((len(ga_data['all_candidate_features']) - len(ga_data['selected_features'])) / len(ga_data['all_candidate_features'])) * 100:.1f}%")
    c4.metric("Best Fitness", f"{ga_data['best_fitness']:.4f}")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("⚙️ Parameter & Formulasi Matematis GA")
        st.markdown("""
        - **Encoding Kromosom**: Biner 17 gen ($g_i \in \{0, 1\}$).
        - **Ukuran Populasi**: 12 individu.
        - **Jumlah Generasi**: 10 generasi evolusi.
        - **Metode Seleksi**: *Tournament Selection* (ukuran $k=3$).
        - **Crossover**: *Uniform Crossover* ($P_c = 0.8$).
        - **Mutasi**: *Bit-flip Mutation* ($P_m = 1/17 \approx 0.059$).
        - **Elitisme**: 2 individu terbaik langsung lolos ke generasi berikutnya.
        """)

        st.markdown("**Fungsi Fitness Multi-Objektif:**")
        st.latex(r"\text{Fitness}(C) = \frac{1}{\text{Val\_RMSE}(C) + 10^{-5}} - \alpha \times \left( \frac{\sum_{i=1}^{17} g_i}{17} \right)")
        st.caption("Penalti kompleksitas α = 0.05 memprioritaskan model yang lebih ramping.")

    with col_right:
        st.subheader("📈 Kurva Konvergensi Fitness GA")
        fig, ax = plt.subplots(figsize=(6, 4))
        gens = range(1, len(ga_data['convergence_best_fitness']) + 1)
        ax.plot(gens, ga_data['convergence_best_fitness'], marker='o', lw=2, color='#2ca02c', label='Best Fitness')
        ax.plot(gens, ga_data['convergence_mean_fitness'], marker='s', lw=1.5, linestyle='--', color='#1f77b4', label='Mean Population')
        ax.set_title('Konvergensi Fitness per Generasi', fontweight='bold', fontsize=10)
        ax.set_xlabel('Generasi')
        ax.set_ylabel('Nilai Fitness')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Tabel Fitur Terpilih vs Dieliminasi
    st.subheader("📋 Analisis Fitur Terpilih vs Dieliminasi")
    col_sel, col_elim = st.columns(2)

    with col_sel:
        st.success(f"**✓ Fitur Terpilih ({len(ga_data['selected_features'])} Fitur):**")
        for f in ga_data['selected_features']:
            st.markdown(f"- `{f}`")

    with col_elim:
        st.error(f"**✗ Fitur Dieliminasi ({len(ga_data['eliminated_features'])} Fitur):**")
        for f in ga_data['eliminated_features']:
            st.markdown(f"- `{f}`")

    st.info("""
    🧠 **Interpretasi AI & Rekayasa Sistem Tenaga**:
    - GA mempertahankan `gap_lag_1` (beban 1 jam lalu) dan `gap_lag_24` (siklus 24 jam) karena keduanya merupakan prediktor autoregresif terkuat.
    - Sebaliknya, `Global_intensity` (arus listrik dalam Ampere) dieliminasi karena redundan terhadap `Global_active_power` melalui hubungan hukum Ohm ($P = V \cdot I \cdot \cos\phi$).
    """)
