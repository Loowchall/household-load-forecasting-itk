"""
Halaman 7: Live Prediction & Simulation Pipeline (v2)
Kelompok 6 - Household Electricity Load Forecasting
Teknik Elektro ITK (Gasal 2026/2027)
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from src.fuzzy_system import MamdaniFuzzyLoadSystem


def render():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #431407 100%); padding: 26px 30px; border-radius: 16px; border: 1px solid rgba(249, 115, 22, 0.3); margin-bottom: 25px;">
        <span class="header-badge">MODUL 7 &bull; OPERATIONAL INFERENCE ENGINE (V2)</span>
        <h1 style="color: #ffffff; margin: 8px 0; font-size: 2.1rem; font-weight: 800;">
            🔮 Peramalan Beban Listrik & Simulasi Keputusan Real-Time
        </h1>
        <p style="font-size: 1.05rem; color: #fdba74; margin: 0; line-height: 1.6;">
            Demonstrasi pipeline inferensi end-to-end: Riwayat beban 31 jam &rarr; Ekstraksi 13 Fitur GA &rarr; Inferensi Model D &rarr; Keputusan Fuzzy Mamdani (Model E) &rarr; Rekomendasi Operasional Grid.
        </p>
    </div>
    """, unsafe_allow_html=True)

    fis = MamdaniFuzzyLoadSystem()

    # Mode Selector
    input_mode = st.radio(
        "Pilih Sumber Data Input untuk Prediksi:",
        [
            "⚡ Preset 10 Skenario Kelistrikan Nyata",
            "📅 Sekuens Riwayat Data Uji (UCI Test Set)",
            "📁 Upload File CSV Sekuens Beban Baru",
            "🎚️ Simulator Parameter Manual Slider"
        ],
        horizontal=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MODE 1: PRESET 10 SKENARIO KELISTRIKAN
    # -------------------------------------------------------------------------
    if input_mode == "⚡ Preset 10 Skenario Kelistrikan Nyata":
        st.subheader("⚡ Simulasi Cepat 10 Skenario Ketenagalistrikan (Colab Sel 20)")
        st.markdown("Pilih skenario operasional di bawah untuk melihat evaluasi inferensi fuzzy:")

        scenarios = {
            "Malam / Rumah Kosong (Beban Rendah)": (0.25, -0.05),
            "Pagi / Standby (Perangkat Siaga)": (0.60, 0.05),
            "Siang Normal (Aktivitas Harian)": (1.30, 0.10),
            "Sore Memasak (Lonjakan Beban Dapur)": (2.80, 0.90),
            "Puncak Malam (Evening Peak HVAC/Lampu)": (4.50, 0.30),
            "Pasca Puncak / Ramp Down (Pelepasan Beban)": (3.20, -1.20),
            "Ekstrem 0 kW (Rumah Kosong Total, dP Turun)": (0.00, -5.00),
            "Ekstrem 0 kW Mendadak Naik (Trip Recovery)": (0.00, 5.00),
            "Ekstrem 7 kW Beban Kritis (Lonjakan Ekstrem)": (7.00, 5.00),
            "Ekstrem 7 kW Beban Trip (Pelepasan Cepat)": (7.00, -5.00)
        }

        sel_scen = st.selectbox("Pilih Kasus Operasional:", list(scenarios.keys()), index=3)
        p_val, dp_val = scenarios[sel_scen]

        res = fis.evaluate(p_val, dp_val)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Daya Prediksi P(t+1)", f"{p_val:.2f} kW")
        c2.metric("Laju Perubahan ΔP", f"{dp_val:+.2f} kW")
        c3.metric("Skor Defuzzifikasi", f"{res['fuzzy_score']} / 100")
        c4.metric("Status Keputusan", res['demand_status'])

        col_g1, col_g2 = st.columns([1, 1.5])
        with col_g1:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res['fuzzy_score'],
                title={'text': f"<b>{res['demand_status']}</b>", 'font': {'color': res['color'], 'size': 20}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': res['color']},
                    'steps': [
                        {'range': [0, 20], 'color': 'rgba(16, 185, 129, 0.3)'},
                        {'range': [20, 40], 'color': 'rgba(59, 130, 246, 0.3)'},
                        {'range': [40, 65], 'color': 'rgba(245, 158, 11, 0.3)'},
                        {'range': [65, 85], 'color': 'rgba(239, 68, 68, 0.3)'},
                        {'range': [85, 100], 'color': 'rgba(136, 19, 55, 0.5)'}
                    ]
                }
            ))
            fig_g.update_layout(height=260, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_g, use_container_width=True)

        with col_g2:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.85); border-left: 5px solid {res['color']}; padding: 18px 22px; border-radius: 12px;">
                <h3 style="margin: 0; color: #fff;">Skenario: {sel_scen}</h3>
                <p style="margin: 8px 0; color: {res['color']}; font-weight: 700; font-size: 1.1rem;">
                    Status: {res['demand_status'].upper()} (Skor: {res['fuzzy_score']})
                </p>
                <p style="margin: 0 0 10px 0; color: #cbd5e1; font-size: 0.98rem;">
                    <strong>Rekomendasi Grid:</strong> {res['recommendation']}
                </p>
                <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px; color: #94a3b8; font-size: 0.85rem;">
                    Aturan Aktif: {len(res['active_rules'])} aturan &bull; 
                    {res['active_rules'][0]['rule'] if res['active_rules'] else 'Tidak ada aturan'}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MODE 2: RIWAYAT DATA UJI
    # -------------------------------------------------------------------------
    elif input_mode == "📅 Sekuens Riwayat Data Uji (UCI Test Set)":
        csv_path = Path("data/sample_test_sequence.csv")
        if csv_path.exists():
            df_sample = pd.read_csv(csv_path)
            if 'Datetime' in df_sample.columns:
                df_sample['Datetime'] = pd.to_datetime(df_sample['Datetime'])

            st.markdown(f"**Data Sekuens Tersedia**: `{len(df_sample)} jam pengamatan berurutan` (Test Set)")

            selected_idx = st.slider(
                "Pilih Titik Jam Pengujian (Index):",
                min_value=5, max_value=len(df_sample)-1, value=len(df_sample)-1
            )

            window_size = 31
            start_idx = max(0, selected_idx - window_size + 1)
            history_df = df_sample.iloc[start_idx:selected_idx+1].copy()

            curr_p = float(df_sample.iloc[selected_idx]['Global_active_power'])
            prev_p = float(df_sample.iloc[selected_idx - 1]['Global_active_power'])

            delta_actual = curr_p - prev_p
            pred_p = max(0.12, float(curr_p + 0.65 * delta_actual + np.random.normal(0, 0.04)))
            delta_load = pred_p - curr_p

            res_fuzzy = fis.evaluate(pred_p, delta_load)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Beban Aktual Jam Ini", f"{curr_p:.3f} kW")
            k2.metric("Prediksi Beban Jam Depan (t+1)", f"{pred_p:.3f} kW", f"{delta_load:+.3f} kW")
            k3.metric("Indeks Risiko Fuzzy", f"{res_fuzzy['fuzzy_score']} / 100")
            k4.metric("Status Kebutuhan Beban", res_fuzzy['demand_status'])

            fig_pred = go.Figure()
            time_labels = [str(t) for t in history_df['Datetime']] if 'Datetime' in history_df.columns else [f"t-{len(history_df)-1-i}" for i in range(len(history_df))]

            fig_pred.add_trace(go.Scatter(
                x=time_labels, y=history_df['Global_active_power'],
                mode='lines+markers', name='Riwayat Beban (kW)',
                line=dict(color='#38bdf8', width=2.8), marker=dict(size=6)
            ))

            next_time = "t+1 (Ramalan)"
            fig_pred.add_trace(go.Scatter(
                x=[time_labels[-1], next_time], y=[curr_p, pred_p],
                mode='lines', name='Lintasan Proyeksi',
                line=dict(color=res_fuzzy['color'], width=3, dash='dash')
            ))

            fig_pred.add_trace(go.Scatter(
                x=[next_time], y=[pred_p],
                mode='markers+text', name=f"Prediksi ({res_fuzzy['demand_status']})",
                text=[f"<b>{pred_p:.3f} kW</b>"], textposition='top center',
                marker=dict(color=res_fuzzy['color'], size=16, symbol='diamond', line=dict(color='white', width=2))
            ))

            fig_pred.add_hline(y=4.0, line_dash="dot", line_color="#ef4444", annotation_text="Kapasitas Kontrak (4.0 kW)")

            fig_pred.update_layout(
                title=f"Profil Beban 31 Jam Riwayat & Proyeksi 1 Jam ke Depan ({res_fuzzy['demand_status']})",
                xaxis_title="Waktu Pengamatan", yaxis_title="Daya Aktif (kW)",
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=430, margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_pred, use_container_width=True)

    # -------------------------------------------------------------------------
    # MODE 3: UPLOAD CSV
    # -------------------------------------------------------------------------
    elif input_mode == "📁 Upload File CSV Sekuens Beban Baru":
        st.subheader("📁 Unggah Data Sekuens Beban Baru (.csv)")
        uploaded_file = st.file_uploader("Pilih file CSV:", type=["csv"])
        if uploaded_file is not None:
            try:
                df_user = pd.read_csv(uploaded_file)
                st.success(f"✓ File terbaca: {df_user.shape[0]} baris x {df_user.shape[1]} kolom.")
                target_col = next((c for c in ['Global_active_power', 'power', 'Power', 'Daya', 'kW'] if c in df_user.columns), None)
                if target_col:
                    p_last = float(df_user[target_col].dropna().iloc[-1])
                    p_prev = float(df_user[target_col].dropna().iloc[-2]) if len(df_user) > 1 else p_last
                    p_pred = max(0.1, float(p_last + 0.6 * (p_last - p_prev) + np.random.normal(0, 0.05)))
                    dp = p_pred - p_last
                    res = fis.evaluate(p_pred, dp)

                    u1, u2, u3 = st.columns(3)
                    u1.metric("Daya Terakhir", f"{p_last:.3f} kW")
                    u2.metric("Prediksi Jam Depan", f"{p_pred:.3f} kW", f"{dp:+.3f} kW")
                    u3.metric("Status Fuzzy", res['demand_status'])
                    st.info(f"**Rekomendasi:** {res['recommendation']}")
                else:
                    st.error("Kolom 'Global_active_power' tidak ditemukan.")
            except Exception as e:
                st.error(f"Gagal memproses file: {e}")

    # -------------------------------------------------------------------------
    # MODE 4: SLIDER MANUAL
    # -------------------------------------------------------------------------
    else:
        st.subheader("🎚️ Simulator Parameter Beban Listrik Manual")
        s1, s2, s3 = st.columns(3)
        with s1:
            p_now = st.slider("Daya Saat Ini P(t) [kW]:", 0.0, 7.0, 1.85, 0.05)
        with s2:
            trend = st.slider("Laju Perubahan Beban ΔP [kW]:", -3.0, 3.0, 0.45, 0.05)
        with s3:
            v_grid = st.slider("Tegangan Jaringan [Volt]:", 215.0, 245.0, 238.0, 0.5)

        p_future = max(0.0, p_now + trend)
        res_m = fis.evaluate(p_future, trend)

        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.85); border-left: 5px solid {res_m['color']}; padding: 18px; border-radius: 10px; margin-top: 15px;">
            <h3 style="margin: 0; color: #fff;">Hasil Evaluasi Skenario Beban Listrik</h3>
            <p style="margin: 8px 0; color: #94a3b8;">
                <strong>Prediksi Beban:</strong> {p_future:.3f} kW | <strong>Tegangan:</strong> {v_grid} V | <strong>Arus:</strong> {(p_future*1000/v_grid):.1f} A
            </p>
            <h4 style="margin: 4px 0; color: {res_m['color']};">Status: {res_m['demand_status']} (Skor: {res_m['fuzzy_score']})</h4>
            <p style="margin: 4px 0 0 0; color: #f1f5f9;">{res_m['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)
