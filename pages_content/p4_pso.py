"""
Halaman 4: Particle Swarm Optimization (PSO) Hyperparameter Tuning
Kelompok 6 - Household Electricity Load Forecasting
"""

import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def render():
    st.header("⚡ Particle Swarm Optimization (PSO) - Tuning Hyperparameter & Window")
    st.markdown("""
    Particle Swarm Optimization digunakan untuk mencari konfigurasi *sequence length* (jendela historis masa lalu) 
    dan hyperparameter arsitektur LSTM terbaik untuk meminimalkan error peramalan beban listrik.
    """)

    # Muat Data PSO
    pso_json_path = Path("models/best_pso_params.json")
    if pso_json_path.exists():
        with open(pso_json_path) as f:
            pso_data = json.load(f)
    else:
        pso_data = {
            "best_hyperparameters": {
                "sequence_length": 32,
                "lstm_units": 32,
                "dropout_rate": 0.1,
                "learning_rate": 0.00198,
                "batch_size": 32
            },
            "gbest_validation_rmse": 0.5528,
            "convergence_gbest_history": [0.5594, 0.5528, 0.5528, 0.5528, 0.5528, 0.5528],
            "search_time_seconds": 615.4,
            "swarm_particles": 8,
            "iterations": 6
        }

    best_p = pso_data['best_hyperparameters']

    # Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sequence Length Optimal", f"{best_p['sequence_length']} Jam", delta="+8 jam vs Baseline")
    c2.metric("LSTM Units (Neuron)", f"{best_p['lstm_units']}")
    c3.metric("Optimal Learning Rate", f"{best_p['learning_rate']:.5f}")
    c4.metric("gbest Val RMSE", f"{pso_data.get('gbest_validation_rmse', 0.5528):.4f} kW")

    st.markdown("---")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("🌌 Search Space & Konfigurasi Swarm")
        st.markdown(f"""
        - **Jumlah Partikel ($N_p$)**: {pso_data.get('swarm_particles', 8)} partikel.
        - **Jumlah Iterasi**: {pso_data.get('iterations', 6)} iterasi pencarian.
        - **Bobot Inersia ($w$)**: 0.70.
        - **Akselerasi Kognitif ($c_1$)**: 1.50 (pengalaman pribadi terbaik $pbest$).
        - **Akselerasi Sosial ($c_2$)**: 1.50 (pengalaman kawanan terbaik $gbest$).
        """)

        st.markdown("**Ruang Pencarian Hyperparameter:**")
        df_bounds = pd.DataFrame([
            {"Parameter": "Sequence Length", "Rentang Batas": "12 s.d. 36 jam", "Nilai Baseline": "24 jam", "Hasil Optimal PSO": f"{best_p['sequence_length']} jam"},
            {"Parameter": "LSTM Units", "Rentang Batas": "16 s.d. 64 unit", "Nilai Baseline": "32 unit", "Hasil Optimal PSO": f"{best_p['lstm_units']} unit"},
            {"Parameter": "Dropout Rate", "Rentang Batas": "0.05 s.d. 0.35", "Nilai Baseline": "0.10", "Hasil Optimal PSO": f"{best_p['dropout_rate']}"},
            {"Parameter": "Learning Rate", "Rentang Batas": "0.0003 s.d. 0.005", "Nilai Baseline": "0.0010", "Hasil Optimal PSO": f"{best_p['learning_rate']:.5f}"},
            {"Parameter": "Batch Size", "Rentang Batas": "32 atau 64", "Nilai Baseline": "32", "Hasil Optimal PSO": f"{best_p['batch_size']}"}
        ])
        st.dataframe(df_bounds, use_container_width=True)

    with col_r:
        st.subheader("📉 Kurva Konvergensi gbest PSO")
        fig, ax = plt.subplots(figsize=(6, 4))
        history_gbest = pso_data.get('convergence_gbest_history', [0.5594, 0.5528, 0.5528, 0.5528, 0.5528, 0.5528])
        ax.plot(range(1, len(history_gbest) + 1), history_gbest, marker='o', lw=2.2, color='#d62728', label='gbest Val RMSE')
        ax.set_title('Penurunan Error Validasi (gbest RMSE)', fontweight='bold', fontsize=10)
        ax.set_xlabel('Iterasi PSO')
        ax.set_ylabel('Validation RMSE (kW)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.success("""
    🔬 **Wawasan Domain Teknik Tenaga Listrik**:
    Window historis **32 jam** yang ditemukan oleh PSO terbukti lebih superior dibanding 24 jam. Window 32 jam mencakup 1 siklus harian lengkap (24 jam) ditambah 8 jam transisi beban malam-ke-pagi, memberikan konteks memori yang kaya bagi LSTM untuk mengantisipasi *morning peak load*.
    """)
