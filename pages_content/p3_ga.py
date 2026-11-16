"""
Halaman 3: Genetic Algorithm (GA) Optimization Visualizer (v2)
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
    <div style="background: linear-gradient(135deg, #0f172a 0%, #064e3b 50%, #022c22 100%); padding: 26px 30px; border-radius: 16px; border: 1px solid rgba(16, 185, 129, 0.3); margin-bottom: 25px;">
        <span class="header-badge">MODUL 3 &bull; METAHEURISTIC FEATURE SELECTION</span>
        <h1 style="color: #ffffff; margin: 8px 0; font-size: 2.1rem; font-weight: 800;">
            🧬 Genetic Algorithm (GA) - Seleksi Fitur & Lag Beban
        </h1>
        <p style="font-size: 1.05rem; color: #a7f3d0; margin: 0; line-height: 1.6;">
            Optimasi multi-seed metaheuristik untuk menyeleksi subset variabel listrik dan lag waktu optimal dengan penalti parsimoni: <strong>13 dari 18 fitur terpilih</strong>, memangkas 5 variabel redundan/berisik dan menghemat parameter LSTM dari 7.073 menjadi 6.433.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Muat Data Hasil GA
    ga_json_path = Path("models/selected_features_ga.json")
    if ga_json_path.exists():
        with open(ga_json_path) as f:
            ga_data = json.load(f)
    else:
        ga_data = {
            "all_features": ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Sub_metering_remainder", "gap_lag_1", "gap_lag_2", "gap_lag_3", "gap_lag_24", "rolling_mean_24h", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"],
            "selected_features": ["Global_active_power", "Global_reactive_power", "Global_intensity", "Sub_metering_3", "Sub_metering_remainder", "gap_lag_2", "gap_lag_24", "rolling_mean_24h", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"],
            "pruned_features": ["Voltage", "Sub_metering_1", "Sub_metering_2", "gap_lag_1", "gap_lag_3"],
            "best_chromosome": [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            "best_fitness": -0.6017,
            "best_val_rmse": 0.5873,
            "generations_run": 8,
            "selected_seed": 7
        }

    # Metric Header Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Kandidat Fitur Awal", len(ga_data.get('all_features', [])), "18 Variabel Input")
    m2.metric("Fitur Terpilih (GA)", len(ga_data.get('selected_features', [])), delta="-5 Fitur Redundan")
    m3.metric("Best Val RMSE", f"{ga_data.get('best_val_rmse', 0.5873):.4f} kW", "Proxy Evaluator")
    m4.metric("Best Fitness Terpilih", f"{ga_data.get('best_fitness', -0.6017):.4f}", f"Seed {ga_data.get('selected_seed', 7)}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Kurva Konvergensi & Multi-Seed (Plotly)",
        "📊 Daftar 13 Fitur Terpilih vs 5 Dieliminasi",
        "📋 Tabel Runs & Frekuensi Antar-Seed",
        "🖼️ Grafik Asli Google Colab (ga_convergence.png)"
    ])

    with tab1:
        st.subheader("Kurva Konvergensi Fitness Multi-Seed (Seed 42 vs Seed 7)")
        st.markdown("""
        Dalam pembaruan v2, GA dijalankan dengan 2 seed independen (Seed 42 dan Seed 7) dengan populasi 10 individu, batas 8 generasi, dan mekanisme *early stopping* (patience=4):
        """)

        # Konvergensi aktual Colab v2
        gens_42 = [1, 2, 3, 4, 5]
        best_fit_42 = [-0.6089, -0.6089, -0.6089, -0.6089, -0.6089]
        mean_fit_42 = [-0.6412, -0.6211, -0.6238, -0.6219, -0.6237]

        gens_7 = [1, 2, 3, 4, 5, 6, 7, 8]
        best_fit_7 = [-0.6101, -0.6101, -0.6101, -0.6101, -0.6017, -0.6017, -0.6017, -0.6017]
        mean_fit_7 = [-0.6526, -0.6342, -0.6224, -0.6244, -0.6161, -0.6215, -0.6160, -0.6226]

        fig_ga = go.Figure()

        # Seed 7 (Selected Run)
        fig_ga.add_trace(go.Scatter(
            x=gens_7, y=best_fit_7,
            mode='lines+markers',
            name='★ Seed 7 - Best Fitness (Dipilih: 13 Fitur)',
            line=dict(color='#00e676', width=3),
            marker=dict(size=9, symbol='star', color='#00e676')
        ))
        fig_ga.add_trace(go.Scatter(
            x=gens_7, y=mean_fit_7,
            mode='lines',
            name='Seed 7 - Rata-rata Populasi',
            line=dict(color='#6ee7b7', width=2, dash='dot')
        ))

        # Seed 42
        fig_ga.add_trace(go.Scatter(
            x=gens_42, y=best_fit_42,
            mode='lines+markers',
            name='Seed 42 - Best Fitness (7 Fitur, Stop Gen 5)',
            line=dict(color='#f59e0b', width=2.5),
            marker=dict(size=7, symbol='circle', color='#f59e0b')
        ))
        fig_ga.add_trace(go.Scatter(
            x=gens_42, y=mean_fit_42,
            mode='lines',
            name='Seed 42 - Rata-rata Populasi',
            line=dict(color='#fcd34d', width=2, dash='dot')
        ))

        fig_ga.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            paper_bgcolor='rgba(15, 23, 42, 0.6)',
            title="Pelacakan Evolusi Fitness: Best vs Mean antar Generasi (Penalti Parsimoni λ=0.02)",
            xaxis=dict(title="Generasi (Gen)", tickmode='linear', tick0=1, dtick=1),
            yaxis_title="Fitness: - (RMSE_val + λ * k)",
            hovermode='x unified',
            height=430,
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_ga, use_container_width=True)

        st.caption("Fungsi Fitness: $\\text{Fitness} = -(\\text{RMSE}_{\\text{val}} + \\lambda \\times k)$, dengan penalti kompleksitas $\\lambda = 0.02$ dan $k$ adalah jumlah fitur aktif.")

    with tab2:
        st.subheader("Distribusi Fitur: 13 Terpilih vs 5 Dieliminasi")

        all_feats = ga_data.get('all_features', [])
        sel_feats = ga_data.get('selected_features', [])

        status_list = ["Terpilih (Aktif)" if f in sel_feats else "Dieliminasi (Redundan)" for f in all_feats]
        colors_list = ["#00e676" if f in sel_feats else "#ef4444" for f in all_feats]

        fig_bars = go.Figure(go.Bar(
            y=all_feats,
            x=[1]*len(all_feats),
            orientation='h',
            marker_color=colors_list,
            text=status_list,
            textposition='inside',
            insidetextanchor='middle',
            hovertemplate="Fitur: <b>%{y}</b><br>Status: %{text}<extra></extra>"
        ))

        fig_bars.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            paper_bgcolor='rgba(15, 23, 42, 0.6)',
            title="Status Seleksi 18 Variabel Input oleh Genetic Algorithm",
            xaxis=dict(visible=False),
            yaxis=dict(autorange="reversed"),
            height=520,
            margin=dict(l=40, r=40, t=50, b=30)
        )
        st.plotly_chart(fig_bars, use_container_width=True)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.success(f"✓ **13 Fitur Terpilih ({len(sel_feats)}/18)**:\n" + ", ".join([f"`{f}`" for f in sel_feats]))
        with col_f2:
            st.error(f"✗ **5 Fitur Dieliminasi (5/18)**:\n" + ", ".join([f"`{f}`" for f in ga_data.get('pruned_features', [])]))

    with tab3:
        st.subheader("📋 Tabel Eksekusi Run GA & Frekuensi Seleksi Antar-Seed")

        r_path = Path("models/ga_runs_table.csv")
        if r_path.exists():
            df_runs = pd.read_csv(r_path)
            st.markdown("**1. Tabel Perbandingan Run Multi-Seed GA:**")
            st.dataframe(df_runs, use_container_width=True)

        freq_path = Path("models/ga_feature_freq_table.csv")
        if freq_path.exists():
            df_freq = pd.read_csv(freq_path)
            st.markdown("**2. Tabel Frekuensi Pemilihan Fitur Antar-Seed (18 Fitur):**")
            st.dataframe(df_freq, use_container_width=True)

        # Representasi Kromosom Biner
        st.markdown("**3. Representasi Kromosom Biner Terbaik (Seed 7):**")
        chrom = ga_data.get('best_chromosome', [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1])
        chrom_cols = st.columns(18)
        for i, (col, bit, name) in enumerate(zip(chrom_cols, chrom, all_feats)):
            with col:
                st.markdown(f"""
                <div style="background: {'rgba(0, 230, 118, 0.25)' if bit==1 else 'rgba(239, 68, 68, 0.25)'}; 
                            border: 2px solid {'#00e676' if bit==1 else '#ef4444'}; 
                            border-radius: 6px; text-align: center; padding: 6px 0;">
                    <span style="font-weight: 800; color: {'#00e676' if bit==1 else '#ef4444'}; font-size: 1.1rem;">{bit}</span>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"<div style='font-size: 0.65rem; text-align: center;'>G{i+1}</div>", unsafe_allow_html=True)

    with tab4:
        st.subheader("🖼️ Grafik Konvergensi Asli Google Colab (models/figures/ga_convergence.png)")
        p_ga_colab = Path("models/figures/ga_convergence.png")
        if p_ga_colab.exists():
            st.image(str(p_ga_colab), caption="Grafik Konvergensi Fitness Best/Mean & Frekuensi Fitur Antar-Seed dari Colab (Sel 16)", use_container_width=True)
        else:
            st.info("File gambar artefak ga_convergence.png siap dimuat.")
