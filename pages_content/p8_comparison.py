"""
Halaman 8: Model Comparison & Benchmark Eksperimen
Kelompok 6 - Household Electricity Load Forecasting
"""

import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def render():
    st.header("⚖️ Komparasi Empat Konfigurasi Model (Benchmark Wajib Dosen)")
    st.markdown("""
    Evaluasi empiris perbandingan performa antara model acuan (*Baseline*), model optimasi tunggal (GA dan PSO), 
    serta model integrasi hibrida ganda (**GA + PSO + LSTM**) pada Test Set kronologis yang belum pernah dilihat.
    """)

    # Muat Data Komparasi Lengkap
    comp_path = Path("models/all_models_comparison.json")
    if comp_path.exists():
        with open(comp_path) as f:
            comp_data = json.load(f)
        df_comp = pd.DataFrame(comp_data)
    else:
        df_comp = pd.DataFrame([
            {"Model": "Model A (Baseline LSTM)", "Fitur": 18, "Window": "24h", "Units": 32, "LR": "0.0010", "MAE (kW)": 0.3263, "RMSE (kW)": 0.4646, "R2 Score": 0.5830, "Waktu (s)": 221.06},
            {"Model": "Model B (GA + LSTM)", "Fitur": 10, "Window": "24h", "Units": 32, "LR": "0.0010", "MAE (kW)": 0.3351, "RMSE (kW)": 0.4691, "R2 Score": 0.5750, "Waktu (s)": 198.98},
            {"Model": "Model C (PSO + LSTM)", "Fitur": 18, "Window": "32h", "Units": 32, "LR": "0.0020", "MAE (kW)": 0.3267, "RMSE (kW)": 0.4598, "R2 Score": 0.5919, "Waktu (s)": 261.59},
            {"Model": "Model D (GA + PSO + LSTM)", "Fitur": 10, "Window": "32h", "Units": 32, "LR": "0.0020", "MAE (kW)": 0.3169, "RMSE (kW)": 0.4616, "R2 Score": 0.5887, "Waktu (s)": 245.84}
        ])

    st.subheader("📋 Tabel Rekapitulasi Metrik Evaluasi Test Set")
    st.dataframe(df_comp, use_container_width=True)

    st.markdown("---")

    # Grafik Komparasi Batang
    st.subheader("📊 Visualisasi Grafik Komparasi Metrik")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = ['Model A\n(Baseline)', 'Model B\n(GA)', 'Model C\n(PSO)', 'Model D\n(GA+PSO)']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

    # 1. MAE
    mae_vals = [float(x) for x in df_comp['MAE (kW)']]
    axes[0].bar(labels, mae_vals, color=colors, width=0.55)
    axes[0].set_title('MAE (kW) - Makin Rendah Makin Baik', fontweight='bold', fontsize=11)
    axes[0].set_ylabel('MAE (kW)')
    axes[0].set_ylim(0.28, 0.36)
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)
    for i, v in enumerate(mae_vals):
        axes[0].text(i, v + 0.002, f"{v:.4f}", ha='center', fontweight='bold', fontsize=9.5)

    # 2. RMSE
    rmse_vals = [float(x) for x in df_comp['RMSE (kW)']]
    axes[1].bar(labels, rmse_vals, color=colors, width=0.55)
    axes[1].set_title('RMSE (kW) - Makin Rendah Makin Baik', fontweight='bold', fontsize=11)
    axes[1].set_ylabel('RMSE (kW)')
    axes[1].set_ylim(0.42, 0.49)
    axes[1].grid(axis='y', linestyle=':', alpha=0.6)
    for i, v in enumerate(rmse_vals):
        axes[1].text(i, v + 0.002, f"{v:.4f}", ha='center', fontweight='bold', fontsize=9.5)

    # 3. R2 Score
    r2_col = 'R2 Score' if 'R2 Score' in df_comp.columns else 'R" Score'
    r2_vals = [float(x) for x in df_comp[r2_col]]
    axes[2].bar(labels, r2_vals, color=colors, width=0.55)
    axes[2].set_title('R² Score - Makin Tinggi Makin Baik', fontweight='bold', fontsize=11)
    axes[2].set_ylabel('R² Score')
    axes[2].set_ylim(0.54, 0.62)
    axes[2].grid(axis='y', linestyle=':', alpha=0.6)
    for i, v in enumerate(r2_vals):
        axes[2].text(i, v + 0.002, f"{v:.4f}", ha='center', fontweight='bold', fontsize=9.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Analisis Temuan Ilmiah
    st.subheader("💡 Analisis Pareto-Optimalitas & Refleksi Keterbatasan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Analisis Dampak Genetic Algorithm (Model B):**
        - Memangkas **44.4% fitur input** (dari 18 menjadi 10 fitur).
        - Mempertahankan **98.6% kemampuan prediktif** dengan waktu pelatihan 10% lebih hemat.
        - Membuktikan prinsip parsimoni (*Occam's razor*) bahwa fitur berdimensi lebih ramping mengeliminasi multikolinearitas.

        **2. Analisis Dampak PSO (Model C):**
        - Menemukan bahwa memperluas jendela historis dari 24 jam menjadi **32 jam** memberikan RMSE terendah (0.4598 kW) dan $R^2$ tertinggi (0.5919).
        - Menunjukkan bahwa transisi beban dini hari memberikan konteks prediktif krusial.
        """)

    with col2:
        st.markdown("""
        **3. Mengapa Model D Terpilih sebagai Arsitektur Final?**
        - Model D mencapai **MAE terendah dari seluruh konfigurasi (0.3169 kW = 316.9 Watt)**.
        - Memiliki kekompakan fitur tinggi (hanya 10 fitur) sekaligus memanfaatkan keunggulan jendela 32 jam dari PSO.
        - Merupakan titik kesetimbangan optimal antara akurasi fisik, efisiensi komputasi, dan kepraktisan implementasi di lapangan.

        **4. Keterbatasan Sistem (*System Limitations*):**
        - Prediksi berbasis data historis murni tanpa sensor cuaca eksternal (suhu ambien / kelembapan) yang dapat memengaruhi penggunaan AC.
        - Beban impulsif mendadak dari peralatan berdaya besar yang menyala acak tetap menjadi sumber utama residual error pada RMSE.
        """)
