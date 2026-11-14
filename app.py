"""
Aplikasi Web Dashboard Streamlit: Household Electricity Load Forecasting
Mata Kuliah: Kecerdasan Buatan (TE2514031) - Institut Teknologi Kalimantan
Kelompok 6 - Gasal 2026/2027
Dosen Pengampu: Ir. Muhammad Ridho Dewanto, S.T., M.T.
"""

import streamlit as st

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Household Load Forecasting | Kelompok 6 ITK",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Styling untuk UI/UX Modern & Profesional
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Card Container Modern */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.01));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 230, 118, 0.5);
    }
    
    /* Header Badge */
    .header-badge {
        display: inline-block;
        padding: 5px 14px;
        background: rgba(0, 230, 118, 0.15);
        color: #00e676;
        border: 1px solid rgba(0, 230, 118, 0.4);
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    /* Pipeline Step Box */
    .pipeline-step {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 4px solid #3b82f6;
    }
    .pipeline-step:hover {
        border-left-color: #00e676;
    }

    /* Sidebar Clean Styling */
    [data-testid="stSidebar"] {
        background: #0b0f19;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.2) !important;
        border-color: #3b82f6 !important;
        color: #60a5fa !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# 3. Import Modul Konten 8 Area Wajib
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

# 4. Sidebar Navigasi Modern
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 15px 0;">
        <div style="display: inline-block; background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 12px 16px; border-radius: 12px; margin-bottom: 8px; box-shadow: 0 4px 15px rgba(30, 60, 114, 0.4);">
            <span style="font-size: 1.8rem;">⚡</span>
        </div>
        <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; letter-spacing: 0.5px;">ITK ELECTRICAL AI</h3>
        <p style="font-size: 0.78rem; color: #9ca3af; margin: 4px 0 0 0;">Hybrid Load Forecasting Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("NAVIGASI 8 AREA WAJIB:")

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
    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px;">
        <span style="font-size: 0.75rem; text-transform: uppercase; color: #60a5fa; font-weight: 700; letter-spacing: 0.5px;">Kelompok 6 - Teknik Elektro</span>
        <ul style="font-size: 0.8rem; margin: 8px 0 0 0; padding-left: 18px; line-height: 1.6; color: #d1d5db;">
            <li>Syahrizal S. E. (04231083)</li>
            <li>Ferdi Pratama (04231032)</li>
            <li>M. Umar Alfaradays (04241070)</li>
            <li>M. Rizky Dimas P. (04231061)</li>
        </ul>
        <div style="margin-top: 8px; font-size: 0.72rem; color: #9ca3af; text-align: center;">
            Semester Gasal 2026/2027 | ITK
        </div>
    </div>
    """, unsafe_allow_html=True)

# 5. Routing Halaman
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

# 6. Footer Bersih & Profesional
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 0.82rem; padding: 12px 0;">
    ⚡ <strong>Tugas Besar Kecerdasan Buatan (TE2514031)</strong> &bull; Kelompok 6 &bull; S1 Teknik Elektro, Institut Teknologi Kalimantan (ITK) &bull; Dosen: Ir. Muhammad Ridho Dewanto, S.T., M.T.
</div>
""", unsafe_allow_html=True)
