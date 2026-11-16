"""
Halaman 4: Particle Swarm Optimization (PSO) Hyperparameter Tuning Visualizer (v2)
Kelompok 6 - Household Electricity Load Forecasting
Teknik Elektro ITK (Gasal 2026/2027)
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path


def render():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #451a03 50%, #1e1b4b 100%); padding: 26px 30px; border-radius: 16px; border: 1px solid rgba(245, 158, 11, 0.3); margin-bottom: 25px;">
        <span class="header-badge">MODUL 4 &bull; SWARM INTELLIGENCE HYPERPARAMETER TUNING</span>
        <h1 style="color: #ffffff; margin: 8px 0; font-size: 2.1rem; font-weight: 800;">
            ⚡ Particle Swarm Optimization (PSO) - Tuning Hyperparameter
        </h1>
        <p style="font-size: 1.05rem; color: #fde68a; margin: 0; line-height: 1.6;">
            Optimasi Swarm Intelligence (8 partikel &times; 6 iterasi) untuk mencari panjang sekuens masa lalu (<em>sequence window</em>), kapasitas unit sel LSTM, laju dropout, <em>learning rate</em>, dan <em>batch size</em> optimal pada Model C dan Model D.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Muat Data PSO
    pso_json_path = Path("models/best_pso_params.json")
    if pso_json_path.exists():
        with open(pso_json_path) as f:
            pso_data = json.load(f)
    else:
        pso_data = {
            "pso_C": {"seq_len": 17, "units": 56, "dropout": 0.161, "learning_rate": 0.00373, "batch_size": 32, "best_val_rmse": 0.5627},
            "pso_D": {"seq_len": 31, "units": 32, "dropout": 0.196, "learning_rate": 0.00360, "batch_size": 32, "best_val_rmse": 0.5658}
        }

    p_c = pso_data.get("pso_C", {})
    p_d = pso_data.get("pso_D", {})

    # Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Window Model D (PSO)", f"{p_d.get('seq_len', 31)} Jam", delta="+7 jam vs Baseline")
    c2.metric("Window Model C (PSO)", f"{p_c.get('seq_len', 17)} Jam", delta="-7 jam vs Baseline")
    c3.metric("Units Model C / D", f"{p_c.get('units', 56)} / {p_d.get('units', 32)}")
    c4.metric("Learning Rate Optimal", f"{p_d.get('learning_rate', 0.00360):.5f}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Konvergensi gbest PSO-C & PSO-D (Plotly)",
        "📋 Tabel Hasil Run Swarm & Ruang Pencarian",
        "🕸️ Radar Chart Karakteristik Parameter",
        "🖼️ Grafik Asli Google Colab (pso_C & pso_D.png)"
    ])

    with tab1:
        st.subheader("Kurva Konvergensi Global Best ($g_{\\text{best}}$ RMSE Validasi)")
        st.markdown("Pelacakan penurunan galat validasi terbaik kawanan partikel selama 6 iterasi:")

        # Data iterasi Colab v2
        iters = [1, 2, 3, 4, 5, 6]
        # Seed 42 & 7 histories
        c_seed42 = [0.5742, 0.5678, 0.5645, 0.5627, 0.5627, 0.5627]
        c_seed7  = [0.5790, 0.5710, 0.5662, 0.5644, 0.5644, 0.5644]
        d_seed42 = [0.5820, 0.5750, 0.5712, 0.5691, 0.5691, 0.5691]
        d_seed7  = [0.5780, 0.5715, 0.5670, 0.5658, 0.5658, 0.5658]

        fig_pso = go.Figure()

        # Model C
        fig_pso.add_trace(go.Scatter(
            x=iters, y=c_seed42,
            mode='lines+markers',
            name='★ PSO-C (18 Fitur) - Seed 42: Window 17h, 56 Units (Dipilih)',
            line=dict(color='#06b6d4', width=3),
            marker=dict(size=8, symbol='circle')
        ))
        fig_pso.add_trace(go.Scatter(
            x=iters, y=c_seed7,
            mode='lines',
            name='PSO-C (18 Fitur) - Seed 7: Window 35h, 56 Units',
            line=dict(color='#67e8f9', width=2, dash='dot')
        ))

        # Model D
        fig_pso.add_trace(go.Scatter(
            x=iters, y=d_seed7,
            mode='lines+markers',
            name='★ PSO-D (13 Fitur GA) - Seed 7: Window 31h, 32 Units (Dipilih)',
            line=dict(color='#f59e0b', width=3),
            marker=dict(size=9, symbol='diamond')
        ))
        fig_pso.add_trace(go.Scatter(
            x=iters, y=d_seed42,
            mode='lines',
            name='PSO-D (13 Fitur GA) - Seed 42: Window 15h, 48 Units',
            line=dict(color='#fde68a', width=2, dash='dot')
        ))

        fig_pso.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            paper_bgcolor='rgba(15, 23, 42, 0.6)',
            title="Konvergensi Swarm Global Best RMSE (Proxy Validation Set)",
            xaxis=dict(title="Iterasi Swarm", tickmode='linear', tick0=1, dtick=1),
            yaxis_title="gbest RMSE Validasi (kW)",
            hovermode='x unified',
            height=430,
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pso, use_container_width=True)

        st.info("💡 **Temuan Teknis Penting**: Pada Model D (fitur ramping hasil GA), PSO menemukan jendela waktu **31 jam** (menangkap siklus 24 jam + transisi pagi 7 jam). Pada Model C (seluruh 18 fitur), PSO mengkompensasi dimensi yang lebih banyak dengan jendela **17 jam** namun memperbesar sel ke **56 units**.")

    with tab2:
        st.subheader("📋 Tabel Hasil Run Swarm & Ruang Pencarian (Search Space)")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            pso_c_csv = Path("models/pso_c_runs_table.csv")
            if pso_c_csv.exists():
                st.markdown("**1. Hasil Pencarian PSO untuk Model C (Semua 18 Fitur):**")
                st.dataframe(pd.read_csv(pso_c_csv), use_container_width=True)

        with col_t2:
            pso_d_csv = Path("models/pso_d_runs_table.csv")
            if pso_d_csv.exists():
                st.markdown("**2. Hasil Pencarian PSO untuk Model D (13 Fitur Hasil GA):**")
                st.dataframe(pd.read_csv(pso_d_csv), use_container_width=True)

        st.markdown("**3. Batas Ruang Pencarian Hyperparameter (Search Space Bounds):**")
        df_bounds = pd.DataFrame([
            {"Hyperparameter": "Sequence Length (Window)", "Batas Bawah": 12, "Batas Atas": 36, "Tipe": "Integer (Jam)", "Fungsi Fisis": "Panjang riwayat beban listrik masa lalu"},
            {"Hyperparameter": "LSTM Units", "Batas Bawah": 16, "Batas Atas": 64, "Tipe": "Kelipatan 8", "Fungsi Fisis": "Kapasitas representasi non-linier tersembunyi"},
            {"Hyperparameter": "Dropout Rate", "Batas Bawah": 0.05, "Batas Atas": 0.30, "Tipe": "Float", "Fungsi Fisis": "Regularisasi pencegah overfitting"},
            {"Hyperparameter": "Learning Rate", "Batas Bawah": "5e-4", "Batas Atas": "8e-3", "Tipe": "Log-uniform", "Fungsi Fisis": "Kecepatan penyesuaian bobot optimizer Adam"},
            {"Hyperparameter": "Batch Size", "Batas Bawah": 32, "Batas Atas": 64, "Tipe": "Kategori {32, 64}", "Fungsi Fisis": "Ukuran mini-batch gradient descent"}
        ])
        st.dataframe(df_bounds, use_container_width=True)

    with tab3:
        st.subheader("🕸️ Radar Chart Perbandingan Konfigurasi Model")

        categories = ['Sequence Length (Jam)', 'LSTM Units', 'Dropout (x100)', 'Learning Rate (x10k)', 'Batch Size']

        fig_radar = go.Figure()
        # Baseline
        fig_radar.add_trace(go.Scatterpolar(
            r=[24, 32, 10, 10, 64],
            theta=categories,
            fill='toself',
            name='Baseline (Model A/B)',
            line=dict(color='#94a3b8', width=2)
        ))
        # PSO-C
        fig_radar.add_trace(go.Scatterpolar(
            r=[17, 56, 16.1, 37.3, 32],
            theta=categories,
            fill='toself',
            name='PSO Model C (18 Fitur)',
            line=dict(color='#06b6d4', width=2.5)
        ))
        # PSO-D
        fig_radar.add_trace(go.Scatterpolar(
            r=[31, 32, 19.6, 36.0, 32],
            theta=categories,
            fill='toself',
            name='PSO Model D (13 Fitur GA)',
            line=dict(color='#f59e0b', width=2.5)
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 70]),
                bgcolor='rgba(15, 23, 42, 0.6)'
            ),
            template='plotly_dark',
            paper_bgcolor='rgba(15, 23, 42, 0.6)',
            title="Perbandingan Skala Hyperparameter Antara Baseline vs Model C vs Model D",
            height=450,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab4:
        st.subheader("🖼️ Grafik PSO Asli Google Colab (models/figures)")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            p_c_img = Path("models/figures/pso_C.png")
            if p_c_img.exists():
                st.image(str(p_c_img), caption="Grafik Konvergensi PSO-C (Semua 18 Fitur) dari Colab Sel 21", use_container_width=True)
        with col_p2:
            p_d_img = Path("models/figures/pso_D.png")
            if p_d_img.exists():
                st.image(str(p_d_img), caption="Grafik Konvergensi PSO-D (13 Fitur GA) dari Colab Sel 22", use_container_width=True)
