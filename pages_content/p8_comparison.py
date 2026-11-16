"""
Halaman 8: Model Comparison & Scientific Benchmark (v2)
Kelompok 6 - Household Electricity Load Forecasting
Teknik Elektro ITK (Gasal 2026/2027)
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def render():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%); padding: 26px 30px; border-radius: 16px; border: 1px solid rgba(20, 184, 166, 0.3); margin-bottom: 25px;">
        <span class="header-badge">MODUL 8 &bull; COMPREHENSIVE BENCHMARK (V2)</span>
        <h1 style="color: #ffffff; margin: 8px 0; font-size: 2.1rem; font-weight: 800;">
            ⚖️ Komparasi Model Pembanding & Baseline Naif
        </h1>
        <p style="font-size: 1.05rem; color: #99f6e4; margin: 0; line-height: 1.6;">
            Evaluasi empiris multi-kriteria berbasis multi-seed training: <strong>Model A (Baseline)</strong>, <strong>Model B (GA)</strong>, <strong>Model C (PSO)</strong>, <strong>Model D (GA + PSO)</strong>, dan <strong>Model E (D + Fuzzy)</strong> dibandingkan terhadap <strong>3 Baseline Naif</strong> pada target test set identik (4.858 jam).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4 Model Overview KPI Cards (Data v2)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <span style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Model A (Baseline)</span>
            <h3 style="margin: 4px 0; color: #60a5fa;">RMSE 0.4684 kW</h3>
            <span style="font-size: 0.78rem; color: #93c5fd;">18 Fitur &bull; W=24h &bull; Params 7.073</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <span style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Model B (GA + LSTM)</span>
            <h3 style="margin: 4px 0; color: #34d399;">RMSE 0.4659 kW</h3>
            <span style="font-size: 0.78rem; color: #6ee7b7;">13 Fitur &bull; Params 6.433 (Hemat)</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card" style="border: 2px solid #06b6d4; background: rgba(6, 182, 212, 0.15);">
            <span style="font-size: 0.75rem; color: #67e8f9; text-transform: uppercase; font-weight: 700;">★ Model C (PSO + LSTM)</span>
            <h3 style="margin: 4px 0; color: #06b6d4;">RMSE 0.4645 kW</h3>
            <span style="font-size: 0.78rem; color: #a5f3fc; font-weight: 600;">RMSE Terendah &bull; W=17h, 56 Units</span>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <span style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Model D (GA + PSO + LSTM)</span>
            <h3 style="margin: 4px 0; color: #f59e0b;">RMSE 0.4762 kW</h3>
            <span style="font-size: 0.78rem; color: #fde68a;">13 Fitur &bull; W=31h &bull; Params 6.433</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Tabel Komparasi Lengkap (Model A-D & Naif)",
        "📊 Visualisasi Grafik Batang & Radar MCDA",
        "💡 Analisis Ilmiah & Evaluasi Multi-Seed",
        "🖼️ Grafik Asli Google Colab (compare_*.png)"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: TABEL REKAPITULASI
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("📋 Tabel Rekapitulasi Metrik Evaluasi Test Set (Deliverable Wajib Dosen)")
        comp_csv = Path("models/comparison_table.csv")
        if comp_csv.exists():
            df_comp = pd.read_csv(comp_csv)
            st.dataframe(df_comp, use_container_width=True)

        st.markdown("---")

        st.subheader("🛑 Pembanding Naif Tanpa Model (Batas Bawah Acuan / Dumb Baseline)")
        st.markdown("Untuk menguji apakah jaringan LSTM benar-benar mempelajari pola temporal atau sekadar mereplikasi data terakhir:")
        naive_csv = Path("models/naive_table.csv")
        if naive_csv.exists():
            df_naive = pd.read_csv(naive_csv)
            st.dataframe(df_naive, use_container_width=True)

        st.info("✓ **Hasil Pembuktian Ilmiah**: Naif terbaik (Lag-1) menghasilkan RMSE **0.5900 kW**, sedangkan seluruh model LSTM menghasilkan RMSE **0.4645 - 0.4762 kW** (penurunan galat ~21%). Hal ini membuktikan arsitektur LSTM berhasil menangkap dependensi non-linier kompleks.")

        abl_csv = Path("models/ablation_table.csv")
        if abl_csv.exists():
            st.markdown("---")
            st.subheader("🔬 Tabel Ablasi Sistem Lengkap (Termasuk Model E Fuzzy)")
            st.dataframe(pd.read_csv(abl_csv), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: VISUALISASI PLOTLY
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("📊 Perbandingan Metrik Evaluasi Utama (MAE, RMSE, R² Score)")

        models_names = ['Model A\n(Baseline)', 'Model B\n(GA)', 'Model C\n(PSO)', 'Model D\n(GA+PSO)', 'Naif\n(Lag-1)']
        mae_vals = [0.3214, 0.3208, 0.3204, 0.3324, 0.3904]
        rmse_vals = [0.4684, 0.4659, 0.4645, 0.4762, 0.5900]
        r2_vals = [0.5707, 0.5752, 0.5778, 0.5559, 0.3190]

        fig_bars = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "MAE (kW) &mdash; Makin Rendah Makin Baik",
                "RMSE (kW) &mdash; Makin Rendah Makin Baik",
                "R² Score &mdash; Makin Tinggi Makin Baik"
            )
        )

        colors_palette = ['#3b82f6', '#10b981', '#06b6d4', '#f59e0b', '#64748b']

        fig_bars.add_trace(go.Bar(
            x=models_names, y=mae_vals, marker_color=colors_palette,
            text=[f"{v:.4f}" for v in mae_vals], textposition='outside', name='MAE'
        ), row=1, col=1)

        fig_bars.add_trace(go.Bar(
            x=models_names, y=rmse_vals, marker_color=colors_palette,
            text=[f"{v:.4f}" for v in rmse_vals], textposition='outside', name='RMSE'
        ), row=1, col=2)

        fig_bars.add_trace(go.Bar(
            x=models_names, y=r2_vals, marker_color=colors_palette,
            text=[f"{v:.4f}" for v in r2_vals], textposition='outside', name='R²'
        ), row=1, col=3)

        fig_bars.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False, height=430, margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_bars, use_container_width=True)

        st.markdown("---")

        # Radar Chart MCDA
        st.subheader("🕸️ Radar Chart Multi-Criteria Decision Analysis (MCDA)")
        categories = ['Akurasi MAE', 'Presisi RMSE', 'Kesesuaian R²', 'Kekompakan Fitur', 'Kecepatan Komputasi']

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=[88, 88, 87, 72, 85], theta=categories, fill='toself', name='Model A (Baseline)', line=dict(color='#3b82f6')))
        fig_radar.add_trace(go.Scatterpolar(r=[89, 89, 88, 100, 80], theta=categories, fill='toself', name='Model B (GA)', line=dict(color='#10b981')))
        fig_radar.add_trace(go.Scatterpolar(r=[90, 90, 90, 72, 70], theta=categories, fill='toself', name='★ Model C (PSO)', line=dict(color='#06b6d4', width=3)))
        fig_radar.add_trace(go.Scatterpolar(r=[85, 85, 84, 100, 68], theta=categories, fill='toself', name='Model D (GA+PSO)', line=dict(color='#f59e0b')))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[50, 100]), bgcolor='rgba(0,0,0,0)'),
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
            height=430, margin=dict(l=40, r=40, t=30, b=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: ANALISIS ILMIAH
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("💡 Analisis Ilmiah Hasil Eksperimen (Berdasarkan Evaluasi Colab v2)")

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown(r"""
            **1. Evaluasi Signifikansi Statistik (Multi-Seed):**
            - Selisih RMSE antara Model A, B, C, dan D sangat tipis: **0.4645 s.d. 0.4762 kW (selisih &le; 0.01 kW)**.
            - Selisih ini berada dalam rentang variasi acak (*seed noise*) antar-pelatihan model.
            - Secara statistik ketenagalistrikan, keempat model LSTM memiliki performa akurasi yang sebanding.

            **2. Keunggulan Model B (GA) dalam Parsimoni:**
            - Model B memangkas 5 variabel input (dari 18 menjadi 13 fitur).
            - Mengurangi jumlah bobot terlatih dari **7.073 menjadi 6.433 parameter**.
            - Mencapai RMSE **0.4659 kW** (sedikit lebih baik dari Baseline 0.4684 kW) dengan model yang lebih ringkas.
            """)

        with col_a2:
            st.markdown(r"""
            **3. Mengapa Model C Meraih RMSE Terendah?**
            - PSO pada Model C menemukan konfigurasi sel yang lebih dalam (**56 unit LSTM**) dengan jendela **17 jam**.
            - Kapasitas representasi non-linier yang lebih besar (17.729 parameter) memberikan RMSE test terendah (**0.4645 kW**).

            **4. Mengapa Nilai Tambah Model E (Fuzzy) Bukan pada Akurasi?**
            - FIS Mamdani bertindak sebagai *decision layer*, bukan regresi daya.
            - Model E mentransformasikan angka kW ke 5 status beban operasional dengan kesesuaian **85.7%** (selisih &le; 1 tingkat vs beban aktual).
            """)

    # -------------------------------------------------------------------------
    # TAB 4: ARTEFAK GAMBAR ASLI COLAB
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("🖼️ Grafik Perbandingan Asli dari Google Colab (models/figures)")

        p_bm = Path("models/figures/compare_metrics.png")
        if p_bm.exists():
            st.image(str(p_bm), caption="1. Grafik Batang Komparasi RMSE, MAE, R2 dari Colab Sel 25", use_container_width=True)

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            p_wk = Path("models/figures/compare_actual_vs_pred_week.png")
            if p_wk.exists():
                st.image(str(p_wk), caption="2. Aktual vs Prediksi Selama 1 Minggu Test Set dari Colab Sel 25", use_container_width=True)

        with col_w2:
            p_err = Path("models/figures/compare_error_over_time.png")
            if p_err.exists():
                st.image(str(p_err), caption="3. MAE Mingguan pada Test Set (Error Terhadap Waktu) dari Colab Sel 25", use_container_width=True)
