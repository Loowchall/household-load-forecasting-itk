"""
Halaman 1: Project Overview & Architecture Visualizer (v2)
Kelompok 6 - Household Electricity Load Forecasting
Teknik Elektro ITK (Gasal 2026/2027)
"""

import streamlit as st
from pathlib import Path


def render():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #1e1b4b 100%); padding: 30px 35px; border-radius: 16px; border: 1px solid rgba(59, 130, 246, 0.3); box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-bottom: 25px;">
        <span class="header-badge">TUGAS BESAR KECERDASAN BUATAN &bull; GASAL 2026/2027 &bull; UPDATE V2</span>
        <h1 style="color: #ffffff; margin: 0 0 10px 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.5px;">
            ⚡ Household Electricity Load Forecasting
        </h1>
        <p style="font-size: 1.1rem; color: #93c5fd; margin: 0; line-height: 1.6; max-width: 950px;">
            Sistem Hibrida Cerdas Berbasis Deep Learning & Bio-Inspired Metaheuristics: 
            <strong>GA</strong> (Seleksi Fitur & Lag Multi-Seed) &bull; 
            <strong>PSO</strong> (Tuning Hyperparameter & Sequence Window) &bull; 
            <strong>LSTM</strong> (Mesin Peramalan Beban 1 Jam ke Depan) &bull; 
            <strong>Mamdani Fuzzy Inference</strong> (Sistem Pendukung Keputusan Ketenagalistrikan 5 Status).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metric Header Cards (Berdasarkan Hasil v2 Multi-Seed)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-card">
            <span style="font-size: 0.78rem; color: #9ca3af; text-transform: uppercase;">RMSE Test Terbaik</span>
            <h2 style="margin: 4px 0; color: #00e676; font-size: 1.85rem; font-weight: 800;">0.4645 kW</h2>
            <span style="font-size: 0.75rem; color: #6ee7b7;">Model C (MAE 0.3204 kW)</span>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card">
            <span style="font-size: 0.78rem; color: #9ca3af; text-transform: uppercase;">Keunggulan vs Naif</span>
            <h2 style="margin: 4px 0; color: #38bdf8; font-size: 1.85rem; font-weight: 800;">-21.3% Galat</h2>
            <span style="font-size: 0.75rem; color: #7dd3fc;">RMSE 0.4645 vs Naif 0.5900 kW</span>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card">
            <span style="font-size: 0.78rem; color: #9ca3af; text-transform: uppercase;">Seleksi Fitur GA</span>
            <h2 style="margin: 4px 0; color: #f59e0b; font-size: 1.85rem; font-weight: 800;">13 Fitur</h2>
            <span style="font-size: 0.75rem; color: #fcd34d;">Pangkas 5 Fitur (Params 6.433)</span>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card">
            <span style="font-size: 0.78rem; color: #9ca3af; text-transform: uppercase;">Kesesuaian Fuzzy</span>
            <h2 style="margin: 4px 0; color: #ec4899; font-size: 1.85rem; font-weight: 800;">85.7%</h2>
            <span style="font-size: 0.75rem; color: #f472b6;">Sesuai &le; 1 Tingkat vs Aktual</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Identitas & Deskripsi Singkat Proyek
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("👥 Identitas Tim Pengembang (Kelompok 6)")
        st.markdown("""
        - **Institusi**: S1 Teknik Elektro, Institut Teknologi Kalimantan (ITK)
        - **Mata Kuliah**: Kecerdasan Buatan (TE2514031) - Kelas B
        - **Dosen Pengampu**: Ir. Muhammad Ridho Dewanto, S.T., M.T.

        **Daftar Anggota & Kontribusi Proyek:**
        1. **Syahrizal Setiawan Ernaldi (04231083)** - *Lead AI Engineer & GA Multi-Seed Optimization*
        2. **Ferdi Pratama (04231032)** - *Machine Learning Researcher & PSO Hyperparameter Tuning*
        3. **Muhammad Umar Alfaradays (04241070)** - *Deep Learning Engineer & LSTM Model Training*
        4. **Muhammad Rizky Dimas Pratama (04231061)** - *Fuzzy Decision System & Cloud Deployment*
        """)

    with col_r:
        st.subheader("🎯 Problem Framing & Domain Ketenagalistrikan")
        st.markdown("""
        - **Problem**: Fluktuasi beban listrik rumah tangga non-linier tinggi menuntut akurasi ramalan pendek (*Short-Term Load Forecasting*) untuk mitigasi trip MCB dan efisiensi energi.
        - **Tujuan**: Memprediksi daya aktif 1 jam ke depan ($P_{t+1}$ kW) dan mengklasifikasikan tingkat kebutuhan daya ke 5 level linguistik fuzzy.
        - **Dataset**: UCI ID 235 (*Individual Household Electric Power Consumption*) dengan 2.075.259 baris menit &rarr; 34.565 jam bersih.
        - **Metodologi Anti-Leakage**: Chronological Split berurutan waktu (70% Train, 15% Val, 15% Test) dengan multi-seed verification.
        """)

    st.markdown("---")

    # Interactive Visual Pipeline
    st.subheader("🔄 Alur Pipeline Sistem Hibrida End-to-End (Update v2)")

    steps = [
        ("1. Data Preprocessing & Agregasi", "Membaca 2M+ data menit, interpolasi missing values gap pendek, agregasi per jam (34.565 jam), dan rekayasa fitur Sub-metering remainder (Sub 4).", "#3b82f6"),
        ("2. Chronological Split (Bebas Leakage)", "Pemisahan data berurutan waktu: 70% Train (24.195 jam), 15% Val (5.185 jam), 15% Test (5.185 jam; 4.858 jam target evaluasi). Scaler fit hanya pada Train.", "#6366f1"),
        ("3. Baseline Naif (Pembanding Wajib)", "Mengevaluasi 3 metode naif tanpa model (lag-1 jam, seasonal 24 jam, rolling mean 24 jam) sebagai standar batas bawah performa.", "#64748b"),
        ("4. Genetic Algorithm (Seleksi Fitur)", "Evolusi 2 seed (Seed 42 & 7) dengan penalti parsimoni menyeleksi 13 fitur listrik dan lag terbaik, memangkas parameter dari 7.073 menjadi 6.433.", "#10b981"),
        ("5. Particle Swarm Optimization", "Swarm 8 partikel x 6 iterasi menemukan hyperparameter optimal: Model C (W=17h, 56 units, batch=32) dan Model D (W=31h, 32 units, batch=32).", "#f59e0b"),
        ("6. Stacked LSTM Forecaster", "Pelatihan deep learning 2 seed per konfigurasi dengan Early Stopping (patience=5) untuk stabilitas hasil statistik.", "#8b5cf6"),
        ("7. Mamdani Fuzzy Inference System", "Fuzifikasi input P(t+1) dan ΔP berbasis kuantil empiris train, 15 aturan konsisten tanpa pelanggaran monotonisitas, defuzzifikasi centroid.", "#ec4899"),
        ("8. Dashboard Streamlit & Cloud Deployment", "Antarmuka interaktif 8 modul dengan Plotly modern, simulator skenario beban real-time, dan verifikasi menyeluruh.", "#06b6d4")
    ]

    for title, desc, color in steps:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border-left: 5px solid {color}; padding: 14px 20px; border-radius: 10px; margin-bottom: 12px;">
            <strong style="color: {color}; font-size: 1.05rem;">{title}</strong>
            <p style="margin: 4px 0 0 0; color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # Expandable preview of all Colab figures
    with st.expander("🖼️ Galeri Visualisasi Artefak Google Colab (models/figures)"):
        st.markdown("Seluruh grafik visualisasi hasil eksekusi Google Colab v2 tersimpan sebagai artefak resmi:")
        fig_dir = Path("models/figures")
        if fig_dir.exists():
            fig_files = sorted(list(fig_dir.glob("*.png")))
            cols = st.columns(3)
            for i, f in enumerate(fig_files):
                with cols[i % 3]:
                    st.image(str(f), caption=f.stem, use_container_width=True)
