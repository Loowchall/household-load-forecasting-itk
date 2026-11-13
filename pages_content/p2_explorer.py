"""
Halaman 2: Dataset Explorer & Exploratory Data Analysis (EDA)
Kelompok 6 - Household Electricity Load Forecasting
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def render():
    st.header("📊 Dataset Explorer & Karakteristik Beban Listrik")
    st.markdown("""
    Eksplorasi data runtun waktu konsumsi listrik rumah tangga dari **UCI Machine Learning Repository (ID 235)**.
    Dataset ini mencatat konsumsi energi listrik per menit di sebuah rumah tangga di Sceaux, Prancis, selama hampir 4 tahun.
    """)

    # Muat Metadata Ringkasan
    summary_path = Path("data/dataset_summary.json")
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {
            "total_raw_rows": 2075259,
            "total_hourly_rows": 34565,
            "date_start": "2006-12-17 17:00:00",
            "date_end": "2010-11-26 21:00:00",
            "train_rows": 24195,
            "val_rows": 5185,
            "test_rows": 5185,
            "target_stats": {
                "mean": 1.0933, "std": 0.8963, "min": 0.124, "p25": 0.3423,
                "median": 0.8029, "p75": 1.5858, "p90": 2.338, "max": 6.5605
            }
        }

    # Metric Cards Utama
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Mentah (Menit)", f"{summary['total_raw_rows']:,}")
    c2.metric("Data Bersih (Per Jam)", f"{summary['total_hourly_rows']:,}")
    c3.metric("Rata-rata Beban", f"{summary['target_stats']['mean']:.3f} kW")
    c4.metric("Beban Puncak Maks", f"{summary['target_stats']['max']:.3f} kW")

    st.markdown("---")

    # Tab Pembagian Data dan Statistik Deskriptif
    tab1, tab2, tab3 = st.tabs(["📈 Profil Beban Runtun Waktu", "📑 Statistik Deskriptif & Kuantil", "🔌 Domain Ketenagalistrikan & Sub-metering"])

    with tab1:
        st.subheader("Visualisasi Profil Beban Listrik Rumah Tangga")
        
        # Load sample data
        data_csv_path = Path("data/sample_test_sequence.csv")
        clean_csv_path = Path("data/household_hourly_clean.csv")
        
        if clean_csv_path.exists():
            df_plot = pd.read_csv(clean_csv_path, index_col=0, parse_dates=True)
            # Tampilkan 168 jam (1 minggu) atau biarkan user memilih rentang
            st.markdown("**Sampel 1 Minggu Profil Beban Listrik (168 Jam):**")
            sample_week = df_plot.iloc[-168:]
            
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(sample_week.index, sample_week['Global_active_power'], color='#1f77b4', lw=1.8, label='Daya Aktif (kW)')
            ax.set_ylabel('Daya Aktif (kW)')
            ax.set_title('Fluktuasi Konsumsi Daya Listrik Harian & Siklus Puncak', fontsize=11, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper right')
            plt.xticks(rotation=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("File dataset bersih tidak ditemukan di folder data/.")

    with tab2:
        st.subheader("Distribusi Statistik Daya Aktif (kW)")
        stats = summary['target_stats']
        
        df_stats = pd.DataFrame([
            {"Parameter": "Nilai Minimum", "Nilai (kW)": f"{stats['min']:.4f}", "Interpretasi Sistem Tenaga": "Beban standby malam hari (peralatan minimum aktif)"},
            {"Parameter": "Kuartil Bawah (P25)", "Nilai (kW)": f"{stats['p25']:.4f}", "Interpretasi Sistem Tenaga": "Beban dasar siang saat rumah kosong"},
            {"Parameter": "Median (P50)", "Nilai (kW)": f"{stats['median']:.4f}", "Interpretasi Sistem Tenaga": "Tingkat konsumsi harian tipikal"},
            {"Parameter": "Rata-rata (Mean)", "Nilai (kW)": f"{stats['mean']:.4f}", "Interpretasi Sistem Tenaga": "Konsumsi rata-rata keseluruhan (~1.09 kWh per jam)"},
            {"Parameter": "Kuartil Atas (P75)", "Nilai (kW)": f"{stats['p75']:.4f}", "Interpretasi Sistem Tenaga": "Aktivitas memasak sore dan penerangan"},
            {"Parameter": "Persentil 90 (P90)", "Nilai (kW)": f"{stats['p90']:.4f}", "Interpretasi Sistem Tenaga": "Beban tinggi simultan (AC + mesin cuci)"},
            {"Parameter": "Nilai Maksimum", "Nilai (kW)": f"{stats['max']:.4f}", "Interpretasi Sistem Tenaga": "Beban puncak absolut mendekati trip MCB"}
        ])
        st.table(df_stats)

    with tab3:
        st.subheader("Dekomposisi Sub-metering & Rekonstruksi Energi")
        st.markdown("""
        Dataset memiliki 3 sensor sub-metering dalam satuan **Watt-hour (Wh)**:
        - **Sub-metering 1**: Dapur (*kitchen*): mesin pencuci piring, microwave.
        - **Sub-metering 2**: Ruang cuci (*laundry room*): mesin cuci, pengering, kulkas.
        - **Sub-metering 3**: Pemanas air (*electric water-heater*) dan pendingin ruangan (*AC*).
        - **Sub-metering Remainder (Fitur Rekayasa)**: Konsumsi lampu penerangan dan stopkontak umum yang tidak terukur oleh sensor 1–3:
        """)
        st.latex(r"E_{\text{unmetered}} = \left( Global\_active\_power \times \frac{1000}{60} \right) - \sum_{i=1}^3 Sub\_metering\_i")
