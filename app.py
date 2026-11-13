"""
Aplikasi Web Dashboard Streamlit: Household Electricity Load Forecasting
Mata Kuliah: Kecerdasan Buatan (TE2514031) - Institut Teknologi Kalimantan
Kelompok 6 - Gasal 2026/2027

Dosen Pengampu: Ir. Muhammad Ridho Dewanto, S.T., M.T.
Tim Pengembang:
1. Syahrizal Setiawan Ernaldi (04231083)
2. Ferdi Pratama (04231032)
3. Muhammad Umar Alfaradays (04241070)
4. Muhammad Rizky Dimas Pratama (04231061)
"""

import streamlit as st

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Load Forecasting - Kelompok 6 ITK",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Import Modul Konten 8 Area Wajib
from pages_content import (
    p1_overview,
    p2_explorer,
    p3_ga,
    p4_pso,
    p5_neural_network,
    p6_fuzzy,
    p7_prediction,
    p8_comparison
)

# 3. Sidebar Navigasi Modern
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding-bottom: 15px;">
        <h2 style="color: #1e3c72; margin: 0;">⚡ ITK ELECTRICAL AI</h2>
        <p style="font-size: 0.85rem; color: #555; margin: 0;">Household Load Forecasting System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Navigasi Modul (8 Area Wajib):**")

    menu_options = [
        "📌 1. Project Overview",
        "📊 2. Dataset Explorer",
        "🧬 3. GA Optimization",
        "⚡ 4. PSO Optimization",
        "🧠 5. Neural Network",
        "🎛️ 6. Fuzzy Inference",
        "🔮 7. Prediction / Simulation",
        "⚖️ 8. Model Comparison"
    ]

    selected_menu = st.radio("Pilih Menu:", menu_options, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    **Identitas Kelompok 6:**
    - Syahrizal S. E. (04231083)
    - Ferdi Pratama (04231032)
    - M. Umar Alfaradays (04241070)
    - M. Rizky Dimas P. (04231061)
    
    *Teknik Elektro ITK 2026/2027*
    """)

# 4. Routing Halaman
if selected_menu == "📌 1. Project Overview":
    p1_overview.render()
elif selected_menu == "📊 2. Dataset Explorer":
    p2_explorer.render()
elif selected_menu == "🧬 3. GA Optimization":
    p3_ga.render()
elif selected_menu == "⚡ 4. PSO Optimization":
    p4_pso.render()
elif selected_menu == "🧠 5. Neural Network":
    p5_neural_network.render()
elif selected_menu == "🎛️ 6. Fuzzy Inference":
    p6_fuzzy.render()
elif selected_menu == "🔮 7. Prediction / Simulation":
    p7_prediction.render()
elif selected_menu == "⚖️ 8. Model Comparison":
    p8_comparison.render()

# 5. Footer Bersih
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem; padding: 10px 0;">
    Tugas Besar Kecerdasan Buatan (TE2514031) | Kelompok 6 - Teknik Elektro Institut Teknologi Kalimantan (ITK) | Gasal 2026/2027
</div>
""", unsafe_allow_html=True)
