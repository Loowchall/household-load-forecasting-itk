"""
Halaman 1: Project Overview & Architecture Visualizer
Kelompok 6 - Household Electricity Load Forecasting
"""

import streamlit as st
import plotly.graph_objects as go


def render():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #1e1b4b 100%); padding: 30px 35px; border-radius: 16px; border: 1px solid rgba(59, 130, 246, 0.3); box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-bottom: 25px;">
        <span class="header-badge">TUGAS BESAR KECERDASAN BUATAN &bull; GASAL 2026/2027</span>
        <h1 style="color: #ffffff; margin: 0 0 10px 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.5px;">
            ⚡ Household Electricity Load Forecasting
        </h1>
        <p style="font-size: 1.1rem; color: #93c5fd; margin: 0; line-height: 1.6; max-width: 900px;">
            Sistem Hibrida Cerdas Berbasis Deep Learning & Bio-Inspired Metaheuristics: 
            <strong>GA</strong> (Seleksi Fitur & Lag) &bull; 
            <strong>PSO</strong> (Tuning Hyperparameter & Sequence Window) &bull; 
            <strong>LSTM</strong> (Mesin Peramalan Beban) &bull; 
            <strong>Fuzzy Inference</strong> (Sistem Pendukung Keputusan Ketenagalistrikan).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metric Header Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-card">
            <span style="font-size: 0.78rem; color: #9ca3af; text-transform: uppercase;">Akurasi MAE Terbaik</span>
            <h2 style="margin: 4px 0; color: #00e676; font-size: 1.9rem; font-weight: 800;">0.3169 kW</h2>
            <span style="font-size: 0.75rem; color: #6ee7b7;">~316.9 Watt (Model D)</span>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card">
            <span style="font-size: 0.78rem; color: #9ca3af; text-transform: uppercase;">Reduksi Fitur GA</span>
            <h2 style="margin: 4px 0; color: #38bdf8; font-size: 1.9rem; font-weight: 800;">44.4%</h2>
            <span style="font-size: 0.75rem; color: #7dd3fc;">10 dari 18 Fitur Terpilih</span>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card">
            <span style="font-size: 0.78rem; color: #9ca3af; text-transform: uppercase;">Window Optimal PSO</span>
            <h2 style="margin: 4px 0; color: #f59e0b; font-size: 1.9rem; font-weight: 800;">32 Jam</h2>
            <span style="font-size: 0.75rem; color: #fcd34d;">+8 Jam vs Default Baseline</span>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card">
            <span style="font-size: 0.78rem; color: #9ca3af; text-transform: uppercase;">Level Keputusan Fuzzy</span>
            <h2 style="margin: 4px 0; color: #ec4899; font-size: 1.9rem; font-weight: 800;">5 Status</h2>
            <span style="font-size: 0.75rem; color: #f472b6;">Very Low s.d. Peak Load</span>
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

        **Daftar Anggota & Kontribusi:**
        1. **Syahrizal Setiawan Ernaldi (04231083)** - *Lead AI Engineer & GA Feature Selection*
        2. **Ferdi Pratama (04231032)** - *ML Researcher & PSO Hyperparameter Optimization*
        3. **Muhammad Umar Alfaradays (04241070)** - *Deep Learning Engineer & LSTM Modeling*
        4. **Muhammad Rizky Dimas Pratama (04231061)** - *Fuzzy Decision System & Cloud Deployment*
        """)

    with col_r:
        st.subheader("🎯 Problem Framing & Domain Ketenagalistrikan")
        st.markdown("""
        - **Problem**: Konsumsi listrik rumah tangga memiliki volatilitas tinggi dengan pola musiman dan jam sibuk (*peak hours*).
        - **Tujuan**: Memprediksi daya aktif 1 jam ke depan ($P_{t+1}$ dalam kW) agar penyedia listrik dan pemilik rumah dapat melakukan mitigasi beban berlebih (*load shedding*).
        - **Dataset**: UCI ID 235 (*Individual Household Electric Power Consumption*) dengan 2.075.259 titik data menit diagregasikan ke 34.565 jam.
        - **Evaluasi**: Menggunakan *Chronological Split* (70% Train, 15% Val, 15% Test) tanpa kebocoran data (*zero leakage*).
        """)

    st.markdown("---")

    # Interactive Visual Pipeline
    st.subheader("🔄 Alur Pipeline Sistem Hibrida End-to-End")

    steps = [
        ("1. Data Preprocessing & Agregasi", "Membaca 2M+ data menit, interpolasi missing values '?', agregasi per jam (1h mean), dan rekayasa fitur Unmetered Load (Sub 4).", "#3b82f6"),
        ("2. Chronological Split (Bebas Leakage)", "Pemisahan data berurutan waktu: 70% Train (2006-2009), 15% Val (2009-2010), 15% Test (2010). Scaler fit hanya pada Train.", "#6366f1"),
        ("3. Genetic Algorithm (Seleksi Fitur)", "Evolusi 10 generasi menyeleksi 10 fitur listrik & lag paling informatif dan memangkas 44.4% variabel redundan.", "#10b981"),
        ("4. PSO (Optimasi Hyperparameter)", "Swarm 8 partikel menemukan sequence length terbaik (32 jam), learning rate optimal (0.0020), dan 32 neuron LSTM.", "#f59e0b"),
        ("5. Stacked LSTM (Forecasting Engine)", "Pelatihan model regresi deep learning dengan Early Stopping untuk memprediksi nilai kuantitatif Pt+1 (kW).", "#8b5cf6"),
        ("6. Mamdani Fuzzy Inference (Decision)", "Mentransformasikan output LSTM dan delta perubahan beban (ΔP) menjadi 5 kategori status kebutuhan listrik.", "#ec4899"),
        ("7. Streamlit Cloud Deployment", "Visualisasi interaktif, simulasi real-time, dan evaluasi 4 model pembanding berbasis web publik.", "#06b6d4")
    ]

    for title, desc, color in steps:
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.08); border-left: 5px solid {color}; border-radius: 8px; padding: 12px 18px; margin-bottom: 8px;">
            <strong style="color: {color}; font-size: 0.95rem;">{title}</strong>
            <p style="margin: 4px 0 0 0; color: #9ca3af; font-size: 0.85rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
