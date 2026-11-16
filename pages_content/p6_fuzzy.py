"""
Halaman 6: Fuzzy Inference System (Mamdani Decision Support - v2)
Kelompok 6 - Household Electricity Load Forecasting
Teknik Elektro ITK (Gasal 2026/2027)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from src.fuzzy_system import MamdaniFuzzyLoadSystem


def render():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #064e3b 100%); padding: 26px 30px; border-radius: 16px; border: 1px solid rgba(16, 185, 129, 0.3); margin-bottom: 25px;">
        <span class="header-badge">MODUL 6 &bull; DECISION SUPPORT SYSTEM (UPDATE V2)</span>
        <h1 style="color: #ffffff; margin: 8px 0; font-size: 2.1rem; font-weight: 800;">
            🎛️ Sistem Inferensi Fuzzy Mamdani (5 Kategori)
        </h1>
        <p style="font-size: 1.05rem; color: #a7f3d0; margin: 0; line-height: 1.6;">
            Mengintegrasikan logika samar human-centric untuk mengonversi hasil peramalan kuantitatif ($P_{t+1}$ dan $\Delta P$) menjadi <strong>5 status kebutuhan beban listrik</strong> yang actionable berbasis kuantil empiris dataset: <em>Very Low, Low, Normal, High,</em> hingga <em>Peak Load</em>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    fis = MamdaniFuzzyLoadSystem()

    # Metric Header Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Kategori Status", "5 Tingkat", "Very Low s.d. Peak")
    m2.metric("Basis Aturan", "15 Aturan IF-THEN", "Konsisten Monoton")
    m3.metric("Kesesuaian vs Aktual", "85.7%", "Selisih <= 1 Tingkat")
    m4.metric("Uji Pelanggaran", "0 Pelanggaran", "PASS Konsistensi")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Kurva Keanggotaan Interaktif (Plotly)",
        "🌐 Permukaan Keputusan 3D (Decision Surface)",
        "🧪 Simulator Live & Speedometer",
        "⚡ 10 Skenario Ketenagalistrikan Nyata",
        "📋 Matriks Keputusan & Konsistensi",
        "🖼️ Grafik Asli Google Colab (Membership & Timeline)"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: KURVA KEANGGOTAAN
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("📊 Visualisasi Interaktif Fungsi Keanggotaan Berbasis Kuantil Empiris")
        st.markdown(r"""
        Pada update v2, parameter fungsi keanggotaan diturunkan secara objektif dari kuantil data latih:
        - Pusat $P(t+1)$ (Kuantil 10%, 35%, 60%, 85%, 95%): `[0.261, 0.402, 1.139, 2.030, 2.942] kW`
        - Skala $\Delta P$ (Kuantil-70 $|\Delta P|$): `0.585 kW`
        """)

        fig_mf = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                "Input 1: Prediksi Daya Aktif P(t+1) [kW] &mdash; 5 Kategori",
                "Input 2: Laju Perubahan Beban &Delta;P [kW] &mdash; 3 Kategori",
                "Output: Indeks Status Kebutuhan Beban Listrik [0..100] &mdash; 5 Kategori"
            ),
            vertical_spacing=0.12
        )

        p_colors = {'Very Low': '#10b981', 'Low': '#3b82f6', 'Normal': '#f59e0b', 'High': '#ef4444', 'Peak': '#881337'}
        for cat, func in fis.mf_p.items():
            fig_mf.add_trace(go.Scatter(
                x=fis.p_universe, y=func(fis.p_universe),
                mode='lines', name=f"P: {cat}",
                line=dict(color=p_colors[cat], width=2.5), fill='tozeroy', opacity=0.8
            ), row=1, col=1)

        d_colors = {'Decreasing': '#3b82f6', 'Steady': '#10b981', 'Increasing': '#ef4444'}
        for cat, func in fis.mf_delta.items():
            fig_mf.add_trace(go.Scatter(
                x=fis.delta_universe, y=func(fis.delta_universe),
                mode='lines', name=f"&Delta;P: {cat}",
                line=dict(color=d_colors[cat], width=2.5), fill='tozeroy', opacity=0.8
            ), row=2, col=1)

        out_colors = {'Very Low': '#10b981', 'Low': '#3b82f6', 'Normal': '#f59e0b', 'High': '#ef4444', 'Peak': '#881337'}
        for cat, func in fis.mf_out.items():
            fig_mf.add_trace(go.Scatter(
                x=fis.out_universe, y=func(fis.out_universe),
                mode='lines', name=f"Out: {cat}",
                line=dict(color=out_colors[cat], width=2.5), fill='tozeroy', opacity=0.8
            ), row=3, col=1)

        fig_mf.update_layout(
            height=750, template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified", margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_mf, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: SURFACE 3D
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("🌐 Visualisasi Permukaan Keputusan 3D (Mamdani Control Surface)")
        st.markdown(r"""
        Grafik 3D kontinu memetakan interaksi non-linier antara $P_{t+1}$ dan $\Delta P$ ke Indeks Status Beban melalui defuzzifikasi **Centroid**:
        """)

        p_grid = np.linspace(0.1, 6.0, 25)
        d_grid = np.linspace(-3.0, 3.0, 25)
        Z_grid = np.zeros((len(d_grid), len(p_grid)))

        for i, d in enumerate(d_grid):
            for j, p in enumerate(p_grid):
                Z_grid[i, j] = fis.evaluate(p, d)['fuzzy_score']

        fig_3d = go.Figure(data=[go.Surface(
            x=p_grid, y=d_grid, z=Z_grid, colorscale='Viridis',
            contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True)))
        )])

        fig_3d.update_layout(
            title="Permukaan Kontrol Keputusan Non-Linier Mamdani FIS (v2)",
            scene=dict(
                xaxis_title="Daya Aktif P(t+1) [kW]",
                yaxis_title="Laju Perubahan &Delta;P [kW]",
                zaxis_title="Indeks Kebutuhan Beban [0-100]"
            ),
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=50, b=10), height=580
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: SIMULATOR LIVE
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("🧪 Simulator Inferensi Mamdani Interaktif")
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            val_p = st.slider("Daya Aktif Prediksi P(t+1) [kW]:", 0.0, 7.0, 2.80, 0.05)
        with c_s2:
            val_d = st.slider("Laju Perubahan Beban ΔP [kW]:", -3.0, 3.0, 0.90, 0.05)

        res_sim = fis.evaluate(val_p, val_d)

        c_g1, c_g2 = st.columns([1.2, 1.8])
        with c_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res_sim['fuzzy_score'],
                number={'suffix': " / 100", 'font': {'size': 26, 'color': "white"}},
                title={'text': f"<b>{res_sim['demand_status'].upper()}</b>", 'font': {'size': 20, 'color': res_sim['color']}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': res_sim['color'], 'thickness': 0.3},
                    'steps': [
                        {'range': [0, 20], 'color': 'rgba(16, 185, 129, 0.4)'},
                        {'range': [20, 40], 'color': 'rgba(59, 130, 246, 0.4)'},
                        {'range': [40, 65], 'color': 'rgba(245, 158, 11, 0.4)'},
                        {'range': [65, 85], 'color': 'rgba(239, 68, 68, 0.4)'},
                        {'range': [85, 100], 'color': 'rgba(136, 19, 55, 0.6)'}
                    ]
                }
            ))
            fig_gauge.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with c_g2:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.7); border-left: 5px solid {res_sim['color']}; padding: 18px 22px; border-radius: 12px;">
                <span style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8;">Hasil Evaluasi Sistem Tenaga Listrik</span>
                <h3 style="margin: 5px 0; color: #ffffff;">Status: <span style="color: {res_sim['color']};">{res_sim['demand_status']}</span> (Skor: {res_sim['fuzzy_score']})</h3>
                <p style="margin: 8px 0 0 0; color: #cbd5e1; font-size: 1rem;">
                    <strong>Rekomendasi Operasional:</strong><br>{res_sim['recommendation']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            if res_sim['rule_firings']:
                st.markdown("**Aturan Aktif yang Memengaruhi Keputusan:**")
                for r in res_sim['rule_firings']:
                    st.caption(f"- `{r['rule']}` (Bobot: **{r['weight']}**)")

    # -------------------------------------------------------------------------
    # TAB 4: 10 SKENARIO KELISTRIKAN NYATA
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("⚡ Pengujian pada 10 Skenario Ketenagalistrikan Nyata (Colab Sel 20)")
        st.markdown("Evaluasi menyeluruh logika inferensi pada skenario ekstrem, normal, hingga transisi beban:")

        scen_csv = Path("models/fuzzy_scenarios_table.csv")
        if scen_csv.exists():
            df_scen = pd.read_csv(scen_csv)
            st.dataframe(df_scen, use_container_width=True)

        st.caption("Perhatikan skenario no. 9 (*Pasca puncak ramp down*): daya $P=3.20\\text{ kW}$ tinggi, namun laju $\\Delta P = -1.20\\text{ kW}$ negatif berhasil menurunkan status dari Peak menjadi **High**, membuktikan sistem sensitif terhadap arah tren.")

    # -------------------------------------------------------------------------
    # TAB 5: MATRIKS KEPUTUSAN & KONSISTENSI
    # -------------------------------------------------------------------------
    with tab5:
        st.subheader("📋 Matriks Keputusan (Confusion Matrix) & Uji Monotonisitas")

        c_cm1, c_cm2 = st.columns(2)
        with c_cm1:
            st.markdown("**1. Matriks Kesesuaian Keputusan pada Test Set (Prediksi vs Aktual):**")
            cm_csv = Path("models/fuzzy_confusion_matrix.csv")
            if cm_csv.exists():
                st.dataframe(pd.read_csv(cm_csv), use_container_width=True)
            st.markdown("""
            - **Persis Sama**: 37.6% (vs Naif 34.7%)
            - **Selisih &le; 1 Tingkat**: 85.7% (vs Naif 72.5%)
            """)

        with c_cm2:
            st.markdown("**2. Matriks 15 Aturan IF-THEN Konsisten:**")
            rule_csv = Path("models/fuzzy_rule_table.csv")
            if rule_csv.exists():
                st.dataframe(pd.read_csv(rule_csv), use_container_width=True)

            st.success("✓ **Uji Monotonisitas Konsistensi (Colab Sel 20)**:\n- Terhadap kenaikan $P$: **0 Pelanggaran Nyata**\n- Terhadap kenaikan $\\Delta P$: **0 Pelanggaran Nyata**")

    # -------------------------------------------------------------------------
    # TAB 6: ARTEFAK GAMBAR ASLI COLAB
    # -------------------------------------------------------------------------
    with tab6:
        st.subheader("🖼️ Grafik Fuzzy Asli dari Google Colab (models/figures)")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            p_mf_img = Path("models/figures/fuzzy_membership.png")
            if p_mf_img.exists():
                st.image(str(p_mf_img), caption="Fungsi Keanggotaan P, dP, dan Status dari Colab Sel 20", use_container_width=True)

            p_cm_img = Path("models/figures/fuzzy_decision_confusion.png")
            if p_cm_img.exists():
                st.image(str(p_cm_img), caption="Matriks Keputusan Fuzzy pada Test Set dari Colab Sel 21", use_container_width=True)

        with col_f2:
            p_time_img = Path("models/figures/model_E_timeline.png")
            if p_time_img.exists():
                st.image(str(p_time_img), caption="Timeline 72 Jam Prediksi LSTM + Status Fuzzy (Model E) dari Colab Sel 21", use_container_width=True)
