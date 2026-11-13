"""
Halaman 6: Fuzzy Inference System (Mamdani Decision Support)
Kelompok 6 - Household Electricity Load Forecasting
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.fuzzy_system import MamdaniFuzzyLoadSystem


def render():
    st.header("🎛️ Fuzzy Inference System (FIS Mamdani) - Pendukung Keputusan")
    st.markdown("""
    Sistem keputusan fuzzy mentransformasikan hasil peramalan daya aktif numerik (kW) dari LSTM menjadi 
    **5 kategori status kebutuhan beban listrik** yang mudah diinterpretasikan oleh operator sistem tenaga dan penghuni rumah tangga.
    """)

    fis = MamdaniFuzzyLoadSystem()

    # Tab Pembahasan
    tab1, tab2, tab3 = st.tabs(["📊 Fungsi Keanggotaan (Membership)", "📜 Matriks 15 Aturan IF-THEN", "🧪 Simulator Inferensi Interaktif"])

    with tab1:
        st.subheader("Kurva Fungsi Keanggotaan (Membership Functions)")
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        # 1. Input 1: P
        p_colors = {'Very Low': '#2ca02c', 'Low': '#1f77b4', 'Normal': '#ff7f0e', 'High': '#d62728', 'Peak': '#7f1416'}
        for cat, func in fis.mf_p.items():
            axes[0].plot(fis.p_universe, func(fis.p_universe), label=cat, color=p_colors[cat], lw=2)
        axes[0].set_title('Input 1: Prediksi Daya Aktif P(t+1) [kW]', fontweight='bold', fontsize=10)
        axes[0].set_ylabel('Derajat (μ)')
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend(loc='upper right', fontsize=8)

        # 2. Input 2: Delta
        d_colors = {'Decreasing': '#1f77b4', 'Steady': '#2ca02c', 'Increasing': '#d62728'}
        for cat, func in fis.mf_delta.items():
            axes[1].plot(fis.delta_universe, func(fis.delta_universe), label=cat, color=d_colors[cat], lw=2)
        axes[1].set_title('Input 2: Laju Perubahan Beban ΔP [kW]', fontweight='bold', fontsize=10)
        axes[1].set_ylabel('Derajat (μ)')
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend(loc='upper right', fontsize=8)

        # 3. Output
        out_colors = {'Very Low': '#2ca02c', 'Low': '#1f77b4', 'Normal': '#ff7f0e', 'High': '#d62728', 'Peak': '#7f1416'}
        for cat, func in fis.mf_out.items():
            axes[2].plot(fis.out_universe, func(fis.out_universe), label=cat, color=out_colors[cat], lw=2)
        axes[2].set_title('Output: Indeks Status Beban Listrik [0..100]', fontweight='bold', fontsize=10)
        axes[2].set_ylabel('Derajat (μ)')
        axes[2].grid(True, linestyle=':', alpha=0.6)
        axes[2].legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.subheader("Matriks 15 Aturan IF-THEN (Mamdani Rule Base)")
        st.markdown("""
        Hubungan sebab-akibat dirancang konsisten dengan dinamika operasional sistem tenaga listrik:
        """)
        
        rule_rows = []
        for i, (p_cat, delta_cat, out_cat) in enumerate(fis.rules, 1):
            rule_rows.append({
                "No": f"R{i}",
                "Kondisi Daya Listrik (P)": p_cat,
                "Laju Perubahan (ΔP)": delta_cat,
                "Keputusan Status Beban": out_cat,
                "Logika Sistem Tenaga": f"Beban {p_cat} dengan kecenderungan {delta_cat} menghasilkan status {out_cat}"
            })
        st.dataframe(rule_rows, use_container_width=True)

    with tab3:
        st.subheader("🧪 Simulasi Inferensi Mamdani Live")
        st.markdown("Geser nilai input untuk melihat proses kalkulasi fuzzifikasi, implikasi Min, dan defuzzifikasi Centroid secara langsung:")

        col_in1, col_in2 = st.columns(2)
        with col_in1:
            sim_p = st.slider("Daya Listrik Prediksi P(t+1) [kW]:", 0.0, 6.5, 1.85, 0.05)
        with col_in2:
            sim_delta = st.slider("Laju Perubahan Beban ΔP [kW]:", -1.5, 1.5, 0.40, 0.05)

        # Eksekusi FIS
        sim_res = fis.evaluate(sim_p, sim_delta)

        # Tampilkan Hasil Keputusan
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown(f"""
            <div style="background-color: {sim_res['color']}; padding: 18px; border-radius: 10px; color: white; text-align: center;">
                <span style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Status Kebutuhan Beban</span>
                <h2 style="margin: 5px 0; color: white;">{sim_res['demand_status']}</h2>
                <span style="font-size: 1.1rem; font-weight: 600;">Skor Fuzzy: {sim_res['fuzzy_score']} / 100</span>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"**Rekomendasi Operasional Listrik:**")
            st.info(sim_res['recommendation'])
            
            st.markdown(f"**Aturan yang Aktif ({sim_res['active_rules_count']} Aturan):**")
            for r in sim_res['rule_firings'][:3]:
                st.caption(f"- `{r['rule']}` (Bobot firing: **{r['weight']}**)")
