"""
Halaman 1: Project Overview & Architecture
Kelompok 6 - Household Electricity Load Forecasting
"""

import streamlit as st


def render():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 25px; border-radius: 12px; color: white; margin-bottom: 25px;">
        <h1 style="color: white; margin: 0; font-size: 2.2rem;">⚡ Household Electricity Load Forecasting</h1>
        <p style="font-size: 1.1rem; margin-top: 8px; opacity: 0.9;">
            Sistem Hibrida AI: Genetic Algorithm (GA) + Particle Swarm Optimization (PSO) + LSTM Neural Network + Mamdani Fuzzy Inference System
        </p>
        <span style="background: #27ae60; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem;">
            Tugas Besar Kecerdasan Buatan (TE2514031) - Institut Teknologi Kalimantan
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Identitas Tim & Kelompok
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("👥 Identitas Tim Pengembang (Kelompok 6)")
        st.markdown("""
        - **Mata Kuliah** : Kecerdasan Buatan (TE2514031) - Kelas B
        - **Dosen Pengampu**: Ir. Muhammad Ridho Dewanto, S.T., M.T.
        - **Program Studi**: S1 Teknik Elektro, Institut Teknologi Kalimantan (ITK)
        - **Semester**     : Gasal 2026/2027

        **Anggota Kelompok 6:**
        1. **Syahrizal Setiawan Ernaldi** (NIM: 04231083)
        2. **Ferdi Pratama** (NIM: 04231032)
        3. **Muhammad Umar Alfaradays** (NIM: 04241070)
        4. **Muhammad Rizky Dimas Pratama** (NIM: 04231061)
        """)

    with col2:
        st.subheader("📋 Ringkasan Problem & Dataset")
        st.markdown("""
        - **Topik Proyek**: *Short-Term Load Forecasting* (STLF) konsumsi beban listrik rumah tangga.
        - **Dataset**: *Individual Household Electric Power Consumption* (UCI Machine Learning Repository, ID 235).
        - **Durasi Rekaman**: Desember 2006 – November 2010 (~47 bulan, 2.075.259 menit).
        - **Agregasi Waktu**: 1 Jam (*1-Hour Mean Aggregation* $\implies$ 34.565 jam).
        - **Target Variabel**: Daya aktif rata-rata 1 jam ke depan ($P_{t+1}$ dalam satuan **kW**).
        """)

    st.markdown("---")

    # Diagram Arsitektur Hybrid System
    st.subheader("🏗️ Diagram Alir Sistem Hibrida (Hybrid Intelligent System Pipeline)")
    st.markdown("""
    Sistem dirancang secara modular end-to-end dengan mengintegrasikan optimasi biologis, komputasi swarm, deep learning temporal, dan logika samar pendukung keputusan:
    """)

    st.code("""
[1. Dataset UCI ID 235] (2.075.259 Baris Data Menit)
          │
          ▼
[2. Preprocessing & Aggregation] 
    - Penanganan Missing Values '?' (Linear Interpolation)
    - Agregasi Rata-rata per Jam (1-Hour Mean -> 34.565 Baris)
    - Rekayasa Fitur: Unmetered Energy (Sub_metering_remainder) & Temporal Lags (t-1 s.d. t-24)
          │
          ▼
[3. Chronological Train-Val-Test Split] (70% : 15% : 15% | Bebas Data Leakage)
          │
          ├────────────────────────────────────────┬────────────────────────────────────────┐
          ▼                                        ▼                                        ▼
[Modul GA: Seleksi Fitur]               [Modul PSO: Tuning]                      [Model A: Baseline LSTM]
- Vektor Biner 17 Gen                   - Search Space 5 Hyperparameters         - Seluruh 18 Fitur
- Multi-objective Fitness               - Partikel Swarm Update Velocity/Posisi  - Window 24 Jam
- Terpilih: 10 Fitur Inti (44.4% Pangkas)- Optimal: Seq 32h, LR 0.002, Units 32   - Benchmark Acuan
          │                                        │
          └───────────────────┬────────────────────┘
                              ▼
           [Model D: Hybrid GA + PSO + LSTM Forecaster]
           - 10 Fitur Terseleksi GA + Sequence Length 32 Jam PSO
           - MAE Terendah: 0.3169 kW (316.9 Watt) | R²: 0.5887
                              │
                              ▼ (Prediksi Daya Pt+1 & Laju Perubahan ΔP)
           [Sistem Inferensi Fuzzy Mamdani (Model E)]
           - 5 Kategori Status: Very Low | Low | Normal | High | Peak
           - 15 Aturan IF-THEN Berbasis Distribusi Empiris Listrik
           - Defuzzifikasi Centroid -> Status Kebutuhan Daya & Rekomendasi
                              │
                              ▼
           [Streamlit Interactive Cloud Dashboard & Deployment]
    """, language="text")

    st.info("""
    💡 **Landasan Rekayasa Ketenagalistrikan**:
    Dalam pengoperasian grid modern, peramalan beban per jam (*Hourly Short-Term Load Forecasting*) sangat esensial untuk manajemen *demand response*, penjadwalan pembangkit cadangan, pencegahan *overloading* trafo distribusi, dan optimasi pemakaian energi rumah tangga cerdas (*smart home energy management*).
    """)
