"""
Halaman 2: Dataset Explorer & Exploratory Data Analysis (EDA)
Kelompok 6 - Household Electricity Load Forecasting
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


def render():
    st.header("📊 Dataset Explorer & Karakteristik Beban Listrik")
    st.markdown("""
    Eksplorasi data runtun waktu konsumsi listrik rumah tangga dari **UCI Machine Learning Repository (ID 235)**.
    Dataset mencakup periode 16 Desember 2006 hingga 26 November 2010 (hampir 4 tahun pengamatan kontinu).
    """)

    # Muat Metadata
    summary_path = Path("data/dataset_summary.json")
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {
            "total_raw_rows": 2075259, "total_hourly_rows": 34565,
            "target_stats": {"mean": 1.0933, "std": 0.8963, "min": 0.124, "p25": 0.3423, "median": 0.8029, "p75": 1.5858, "p90": 2.338, "max": 6.5605}
        }

    # Load clean data for interactive plotting
    clean_csv = Path("data/household_hourly_clean.csv")
    if clean_csv.exists():
        df_clean = pd.read_csv(clean_csv, index_col=0, parse_dates=True)
    else:
        # Fallback dummy sample
        df_clean = pd.DataFrame({'Global_active_power': np.random.uniform(0.2, 3.0, 168)})

    # Metric Cards Utama
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Mentah (Menit)", f"{summary['total_raw_rows']:,}")
    c2.metric("Data Bersih (Per Jam)", f"{summary['total_hourly_rows']:,}")
    c3.metric("Rata-rata Beban", f"{summary['target_stats']['mean']:.3f} kW")
    c4.metric("Beban Puncak Maks", f"{summary['target_stats']['max']:.3f} kW")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Runtun Waktu Interaktif (Plotly)",
        "🥧 Proporsi Konsumsi Energi (Sub-metering)",
        "⏰ Pola Beban Diurnal (Jam 00 - 23)",
        "🔥 Matriks Korelasi Variabel Listrik"
    ])

    with tab1:
        st.subheader("Kurva Profil Beban Listrik Runtun Waktu")
        st.markdown("Pilih rentang tampilan grafik interaktif di bawah ini (gunakan kursor untuk *zoom in/out* dan *hover* detail):")

        view_option = st.selectbox(
            "Pilih Jendela Pengamatan:",
            ["Sampel 7 Hari (168 Jam)", "Sampel 30 Hari (720 Jam)", "Sampel 90 Hari (3 Bulan)"]
        )

        n_hours = 168 if "7 Hari" in view_option else (720 if "30 Hari" in view_option else 2160)
        df_slice = df_clean.iloc[-n_hours:].copy()

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=df_slice.index,
            y=df_slice['Global_active_power'],
            mode='lines',
            name='Daya Aktif (kW)',
            line=dict(color='#00e676', width=2),
            hovertemplate="Waktu: %{x}<br>Daya: <b>%{y:.3f} kW</b><extra></extra>"
        ))

        # Tambahkan batas rata-rata
        mean_val = df_slice['Global_active_power'].mean()
        fig_ts.add_hline(y=mean_val, line_dash="dash", line_color="#38bdf8", annotation_text=f"Rata-rata: {mean_val:.2f} kW")

        fig_ts.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            paper_bgcolor='rgba(15, 23, 42, 0.6)',
            title=f"Profil Beban Listrik Rumah Tangga ({view_option})",
            xaxis_title="Waktu (Timestamp)",
            yaxis_title="Daya Aktif (kW)",
            hovermode="x unified",
            height=420,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with tab2:
        st.subheader("Distribusi Konsumsi Energi Berdasarkan Sub-metering")
        st.markdown("""
        Dataset memiliki 3 sensor energi aktif sub-metering (dalam Wh/menit) ditambah fitur rekayasa **Beban Tak Terukur (*Unmetered*)**:
        - **Sub 1 (Dapur)**: Mesin pencuci piring, microwave.
        - **Sub 2 (Laundry)**: Mesin cuci, pengering pakaian, kulkas.
        - **Sub 3 (Pemanas/AC)**: Pemanas air listrik dan pendingin udara.
        - **Sub 4 (Remainder/Tak Terukur)**: Lampu penerangan dan stopkontak umum.
        """)

        # Hitung kontribusi rata-rata energi
        if 'Sub_metering_1' in df_clean.columns:
            sub1_m = df_clean['Sub_metering_1'].mean()
            sub2_m = df_clean['Sub_metering_2'].mean()
            sub3_m = df_clean['Sub_metering_3'].mean()
            sub4_m = df_clean['Sub_metering_remainder'].mean() if 'Sub_metering_remainder' in df_clean.columns else 8.0
        else:
            sub1_m, sub2_m, sub3_m, sub4_m = 1.1, 1.3, 6.4, 9.5

        sub_labels = ['Sub 1: Dapur (Kitchen)', 'Sub 2: Ruang Cuci (Laundry)', 'Sub 3: Pemanas Air & AC', 'Sub 4: Penerangan & Stopkontak (Unmetered)']
        sub_vals = [sub1_m, sub2_m, sub3_m, sub4_m]
        sub_colors = ['#f59e0b', '#3b82f6', '#ec4899', '#00e676']

        col_donut, col_desc = st.columns([1.2, 1])
        with col_donut:
            fig_donut = go.Figure(data=[go.Pie(
                labels=sub_labels,
                values=sub_vals,
                hole=0.45,
                marker=dict(colors=sub_colors),
                textinfo='percent+label',
                textposition='outside'
            )])
            fig_donut.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(15, 23, 42, 0.6)',
                showlegend=False,
                height=380,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_desc:
            st.markdown("### Ringkasan Proporsi Beban:")
            st.markdown(f"""
            - 🔴 **Sub 3 (AC & Pemanas)**: Mengonsumsi **~{sub3_m/(sum(sub_vals))*100:.1f}%** energi terukur. Peralatan termal merupakan kontributor terbesar pada rumah tangga di Eropa.
            - 🟢 **Beban Tak Terukur (Lampu/TV)**: Menyumbang **~{sub4_m/(sum(sub_vals))*100:.1f}%** dari total beban harian.
            - 🔵 **Sub 2 (Mesin Cuci)**: Menyumbang **~{sub2_m/(sum(sub_vals))*100:.1f}%** dengan pola impulsif saat digunakan.
            - 🟡 **Sub 1 (Dapur/Microwave)**: Menyumbang **~{sub1_m/(sum(sub_vals))*100:.1f}%**.
            """)

    with tab3:
        st.subheader("Siklus Diurnal 24 Jam (Analisis Jam Beban Puncak)")
        st.markdown("Visualisasi variasi konsumsi daya listrik berdasarkan jam (00:00 s.d. 23:00) yang memperlihatkan *Morning Peak* dan *Evening Peak*:")

        # Group by hour
        if isinstance(df_clean.index, pd.DatetimeIndex):
            df_hourly_grouped = df_clean.groupby(df_clean.index.hour)['Global_active_power'].agg(['mean', 'std', 'min', 'max']).reset_index()
            df_hourly_grouped.columns = ['Jam', 'Mean', 'Std', 'Min', 'Max']
        else:
            df_hourly_grouped = pd.DataFrame({
                'Jam': list(range(24)),
                'Mean': [0.4, 0.35, 0.3, 0.28, 0.3, 0.45, 0.8, 1.35, 1.4, 1.2, 1.1, 1.05, 1.1, 1.0, 0.95, 0.9, 1.1, 1.45, 1.9, 2.2, 2.1, 1.8, 1.2, 0.7]
            })

        fig_diurnal = go.Figure()
        fig_diurnal.add_trace(go.Bar(
            x=df_hourly_grouped['Jam'],
            y=df_hourly_grouped['Mean'],
            name='Rata-rata Daya (kW)',
            marker=dict(
                color=df_hourly_grouped['Mean'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Daya (kW)")
            ),
            hovertemplate="Jam: %{x}:00<br>Rata-rata: <b>%{y:.3f} kW</b><extra></extra>"
        ))

        fig_diurnal.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            paper_bgcolor='rgba(15, 23, 42, 0.6)',
            title="Profil Beban Rata-rata per Jam dalam Sehari (Diurnal Cycle)",
            xaxis=dict(title="Jam dalam Sehari (0 - 23)", tickmode='linear', tick0=0, dtick=2),
            yaxis_title="Rata-rata Daya Aktif (kW)",
            height=400,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_diurnal, use_container_width=True)

        st.info("💡 **Observasi Sistem Tenaga Listrik**: Puncak beban malam (*Evening Peak*) terjadi pada pukul **18:00 - 22:00** (mencapai > 2.0 kW) saat seluruh lampu, pemanas/AC, dan perangkat hiburan aktif bersamaan.")

    with tab4:
        st.subheader("Matriks Korelasi Pearson Variabel Listrik")
        
        corr_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        valid_cols = [c for c in corr_cols if c in df_clean.columns]
        
        if len(valid_cols) > 2:
            corr_mat = df_clean[valid_cols].corr()
            fig_corr = px.imshow(
                corr_mat,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale="Blues",
                template="plotly_dark",
                title="Heatmap Korelasi Pearson Variabel Kelistrikan"
            )
            fig_corr.update_layout(
                paper_bgcolor='rgba(15, 23, 42, 0.6)',
                plot_bgcolor='rgba(15, 23, 42, 0.6)',
                height=450
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("Perhatikan korelasi antara `Global_active_power` dan `Global_intensity` mendekati 1.00 (multikolinear sempurna). Inilah alasan fisis mengapa GA mengeliminasi `Global_intensity`.")
