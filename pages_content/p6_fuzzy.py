"""
Halaman 6: Fuzzy Inference System (Mamdani Decision Support)
Kelompok 6 - Household Electricity Load Forecasting
Teknik Elektro ITK (Gasal 2026/2027)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.fuzzy_system import MamdaniFuzzyLoadSystem


def render():
    st.markdown(r"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #064e3b 100%); padding: 26px 30px; border-radius: 16px; border: 1px solid rgba(16, 185, 129, 0.3); margin-bottom: 25px;">
        <span class="header-badge">MODUL 6 &bull; DECISION SUPPORT SYSTEM</span>
        <h1 style="color: #ffffff; margin: 8px 0; font-size: 2.1rem; font-weight: 800;">
            🎛️ Sistem Inferensi Fuzzy Mamdani (5 Kategori)
        </h1>
        <p style="font-size: 1.05rem; color: #a7f3d0; margin: 0; line-height: 1.6;">
            Mengintegrasikan logika samar human-centric untuk mengonversi hasil peramalan kuantitatif ($P_{t+1}$ dan $\Delta P$) menjadi <strong>5 status kebutuhan beban listrik</strong> yang actionable: <em>Very Low, Low, Normal, High,</em> hingga <em>Peak Load</em>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    fis = MamdaniFuzzyLoadSystem()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Kurva Fungsi Keanggotaan Interaktif",
        "🌐 Permukaan Keputusan 3D (Decision Surface)",
        "📜 Basis 15 Aturan IF-THEN & Kaidah Grid",
        "🧪 Simulator Inferensi Live & Speedometer"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: KURVA FUNGSI KEANGGOTAAN
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("📊 Visualisasi Interaktif Fungsi Keanggotaan (Membership Functions)")
        st.markdown("""
        Fungsi keanggotaan menggunakan kurva segitiga (*triangular*) dan trapesium (*trapezoidal*) yang dirancang berdasarkan distribusi empiris konsumsi listrik rumah tangga (UCI Dataset ID 235):
        """)

        # Subplot 3 rows for Input 1, Input 2, and Output
        fig_mf = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                "Input 1: Prediksi Daya Aktif P(t+1) [kW] &mdash; 5 Kategori",
                "Input 2: Laju Perubahan Beban &Delta;P [kW] &mdash; 3 Kategori",
                "Output: Indeks Status Kebutuhan Beban Listrik [0..100] &mdash; 5 Kategori"
            ),
            vertical_spacing=0.12
        )

        # 1. Input 1: P
        p_colors = {'Very Low': '#10b981', 'Low': '#3b82f6', 'Normal': '#f59e0b', 'High': '#ef4444', 'Peak': '#881337'}
        for cat, func in fis.mf_p.items():
            y_vals = func(fis.p_universe)
            fig_mf.add_trace(go.Scatter(
                x=fis.p_universe, y=y_vals,
                mode='lines',
                name=f"P: {cat}",
                line=dict(color=p_colors[cat], width=2.5),
                fill='tozeroy',
                fillcolor=p_colors[cat].replace(')', ', 0.12)').replace('rgb', 'rgba') if 'rgba' in p_colors[cat] else None,
                opacity=0.85
            ), row=1, col=1)

        # 2. Input 2: Delta P
        d_colors = {'Decreasing': '#3b82f6', 'Steady': '#10b981', 'Increasing': '#ef4444'}
        for cat, func in fis.mf_delta.items():
            y_vals = func(fis.delta_universe)
            fig_mf.add_trace(go.Scatter(
                x=fis.delta_universe, y=y_vals,
                mode='lines',
                name=f"&Delta;P: {cat}",
                line=dict(color=d_colors[cat], width=2.5),
                fill='tozeroy',
                opacity=0.85
            ), row=2, col=1)

        # 3. Output
        out_colors = {'Very Low': '#10b981', 'Low': '#3b82f6', 'Normal': '#f59e0b', 'High': '#ef4444', 'Peak': '#881337'}
        for cat, func in fis.mf_out.items():
            y_vals = func(fis.out_universe)
            fig_mf.add_trace(go.Scatter(
                x=fis.out_universe, y=y_vals,
                mode='lines',
                name=f"Out: {cat}",
                line=dict(color=out_colors[cat], width=2.5),
                fill='tozeroy',
                opacity=0.85
            ), row=3, col=1)

        fig_mf.update_layout(
            height=780,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        fig_mf.update_xaxes(title_text="Daya Aktif P (kW)", row=1, col=1)
        fig_mf.update_yaxes(title_text="Derajat Keanggotaan (&mu;)", row=1, col=1)
        fig_mf.update_xaxes(title_text="Laju Perubahan &Delta;P (kW)", row=2, col=1)
        fig_mf.update_yaxes(title_text="Derajat Keanggotaan (&mu;)", row=2, col=1)
        fig_mf.update_xaxes(title_text="Indeks Status Beban (0 - 100)", row=3, col=1)
        fig_mf.update_yaxes(title_text="Derajat Keanggotaan (&mu;)", row=3, col=1)

        st.plotly_chart(fig_mf, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: BIDANG PERMUKAAN KEPUTUSAN 3D
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("🌐 Visualisasi Permukaan Keputusan 3D (Mamdani Control Surface)")
        st.markdown(r"""
        Grafik 3D di bawah ini memetakan seluruh ruang keadaan kontinu dari kombinasi $P_{t+1}$ dan $\Delta P$ ke Indeks Status Beban melalui proses defuzzifikasi **Centroid**. Anda dapat memutar, memperbesar (*zoom*), dan menggeser (*pan*) grafik ini secara interaktif:
        """)

        # Compute 28x28 grid
        p_grid = np.linspace(0.1, 6.0, 28)
        d_grid = np.linspace(-1.5, 1.5, 28)
        Z_grid = np.zeros((len(d_grid), len(p_grid)))

        for i, d in enumerate(d_grid):
            for j, p in enumerate(p_grid):
                Z_grid[i, j] = fis.evaluate(p, d)['fuzzy_score']

        fig_3d = go.Figure(data=[go.Surface(
            x=p_grid,
            y=d_grid,
            z=Z_grid,
            colorscale='Viridis',
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))
            ),
            colorbar=dict(title="Indeks Fuzzy")
        )])

        fig_3d.update_layout(
            title="Permukaan Kontrol Keputusan Non-Linier Mamdani FIS",
            scene=dict(
                xaxis_title="Daya Aktif P(t+1) [kW]",
                yaxis_title="Laju Perubahan &Delta;P [kW]",
                zaxis_title="Indeks Kebutuhan Beban [0-100]",
                camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2))
            ),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=50, b=10),
            height=620
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        st.info(r"💡 **Observasi Fisik**: Terlihat kontur permukaan memiliki lereng curam ketika daya tinggi bertemu dengan $\Delta P$ positif (lonjakan tajam), mengantarkan sistem langsung ke zona merah (*Peak Load*). Sebaliknya, pada $\Delta P$ negatif (pelepasan beban), permukaan melandai secara mulus.")

    # -------------------------------------------------------------------------
    # TAB 3: BASIS ATURAN IF-THEN
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("📜 Matriks 15 Aturan IF-THEN (Mamdani Rule Base)")
        st.markdown("""
        Hubungan sebab-akibat dirancang konsisten dengan dinamika operasional sistem tenaga listrik dan proteksi beban lebih:
        """)

        rule_data = []
        for i, (p_cat, delta_cat, out_cat) in enumerate(fis.rules, 1):
            rule_data.append({
                "Aturan": f"R{i}",
                "Kondisi Daya Aktif (P)": p_cat,
                "Laju Perubahan (ΔP)": delta_cat,
                "Konsekuensi Output": out_cat,
                "Implikasi Grid & Beban": f"Jika daya {p_cat} dan tren {delta_cat}, maka status disimpulkan sebagai {out_cat}"
            })
        st.dataframe(pd.DataFrame(rule_data), use_container_width=True, height=450)

    # -------------------------------------------------------------------------
    # TAB 4: SIMULATOR LIVE & GAUGE
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("🧪 Simulator Inferensi Mamdani Interaktif")
        st.markdown("Atur slider input di bawah untuk mengamati proses fuzzifikasi, inferensi min-max, dan defuzzifikasi centroid seketika:")

        c_s1, c_s2 = st.columns(2)
        with c_s1:
            val_p = st.slider("Daya Aktif Prediksi P(t+1) [kW]:", min_value=0.0, max_value=6.5, value=2.65, step=0.05)
        with c_s2:
            val_d = st.slider("Laju Perubahan Beban ΔP [kW]:", min_value=-1.5, max_value=1.5, value=0.45, step=0.05)

        # Evaluasi
        res_sim = fis.evaluate(val_p, val_d)

        st.markdown("---")

        c_g1, c_g2 = st.columns([1.2, 1.8])

        with c_g1:
            # Plotly Indicator Speedometer Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res_sim['fuzzy_score'],
                number={'suffix': " / 100", 'font': {'size': 26, 'color': "white"}},
                title={'text': f"<b>{res_sim['demand_status'].upper()}</b>", 'font': {'size': 20, 'color': res_sim['color']}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': res_sim['color'], 'thickness': 0.3},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 20], 'color': 'rgba(16, 185, 129, 0.4)'},
                        {'range': [20, 40], 'color': 'rgba(59, 130, 246, 0.4)'},
                        {'range': [40, 65], 'color': 'rgba(245, 158, 11, 0.4)'},
                        {'range': [65, 85], 'color': 'rgba(239, 68, 68, 0.4)'},
                        {'range': [85, 100], 'color': 'rgba(136, 19, 55, 0.6)'}
                    ]
                }
            ))
            fig_gauge.update_layout(
                height=320,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with c_g2:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.7); border-left: 5px solid {res_sim['color']}; padding: 18px 22px; border-radius: 12px; margin-bottom: 15px;">
                <span style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 1px;">Hasil Evaluasi Sistem Tenaga</span>
                <h3 style="margin: 5px 0; color: #ffffff;">Status: <span style="color: {res_sim['color']};">{res_sim['demand_status']}</span> (Skor: {res_sim['fuzzy_score']})</h3>
                <p style="margin: 8px 0 0 0; color: #cbd5e1; font-size: 1rem; line-height: 1.5;">
                    <strong>Rekomendasi Operasional:</strong> {res_sim['recommendation']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Bar chart of active rule firings
            if res_sim['rule_firings']:
                active_rules_names = [f"R{i+1}: " + r['rule'].split('THEN')[0].replace('IF ', '') for i, r in enumerate(res_sim['rule_firings'])]
                active_weights = [r['weight'] for r in res_sim['rule_firings']]

                fig_firings = go.Figure(go.Bar(
                    x=active_weights,
                    y=active_rules_names,
                    orientation='h',
                    marker_color=res_sim['color'],
                    text=[f"{w:.2f}" for w in active_weights],
                    textposition='outside'
                ))
                fig_firings.update_layout(
                    title=f"Aturan Aktif yang Memengaruhi Keputusan ({res_sim['active_rules_count']} Aturan):",
                    xaxis_title="Derajat Pembakaran (&alpha;)",
                    xaxis=dict(range=[0, 1.1]),
                    height=200,
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=20, t=35, b=20)
                )
                st.plotly_chart(fig_firings, use_container_width=True)
            else:
                st.info("Tidak ada aturan yang aktif dengan derajat > 0 pada titik ini.")
