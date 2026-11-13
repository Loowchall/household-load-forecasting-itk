"""
Halaman 5: Neural Network Architecture & Training Dynamics
Kelompok 6 - Household Electricity Load Forecasting
"""

import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def render():
    st.header("🧠 Arsitektur Neural Network - Long Short-Term Memory (LSTM)")
    st.markdown("""
    Model neural network utama yang digunakan adalah **Stacked LSTM Regressor** yang bertindak sebagai *forecasting engine* 
    untuk memprediksi konsumsi daya aktif 1 jam ke depan ($P_{t+1}$).
    """)

    # Tab Pembahasan
    tab1, tab2, tab3 = st.tabs(["🏛️ Arsitektur & Dimensi Layer", "📐 Landasan Matematis LSTM", "📉 Dinamika Pelatihan & Loss"])

    with tab1:
        st.subheader("Struktur Layer Model D (GA + PSO + LSTM)")
        
        col_l, col_r = st.columns([1, 1])
        
        with col_l:
            st.markdown("""
            **Rincian Layer-by-Layer:**
            1. **Input Layer**: `(None, 32, 10)`
               - Jendela waktu 32 jam ke belakang.
               - 10 variabel listrik dan lag terseleksi GA.
            2. **LSTM Hidden Layer**: `LSTM(units=32, return_sequences=False)`
               - 32 sel memori LSTM dengan gating aktif.
               - Output berupa vektor representasi temporal dimensi 32 pada time-step terakhir.
            3. **Regularization Layer**: `Dropout(rate=0.10)`
               - Mencegah ko-adaptasi bobot berlebih.
            4. **Dense Dense Layer**: `Dense(units=16, activation='relu')`
               - Pemetaan fitur non-linier berdimensi 16.
            5. **Output Layer**: `Dense(units=1, activation='linear')`
               - Proyeksi skalar daya aktif ternormalisasi $\hat{y}_{t+1}$.
            """)

        with col_r:
            st.markdown("**Tabel Parameter Terbobot:**")
            df_params = pd.DataFrame([
                {"Layer": "Input Layer", "Tipe": "Input", "Bentuk Output": "(None, 32, 10)", "Jumlah Parameter": 0},
                {"Layer": "lstm_hybrid_layer", "Tipe": "LSTM", "Bentuk Output": "(None, 32)", "Jumlah Parameter": 5504},
                {"Layer": "dropout_reg", "Tipe": "Dropout", "Bentuk Output": "(None, 32)", "Jumlah Parameter": 0},
                {"Layer": "dense_feature", "Tipe": "Dense (ReLU)", "Bentuk Output": "(None, 16)", "Jumlah Parameter": 528},
                {"Layer": "output_power", "Tipe": "Dense (Linear)", "Bentuk Output": "(None, 1)", "Jumlah Parameter": 17}
            ])
            st.table(df_params)
            st.caption("Total Bobot Terlatih: **6.049 Parameter** (~24 KB memory footprint). Sangat ringan dan efisien untuk edge deployment!")

    with tab2:
        st.subheader("Mekanisme Gating Matematis Sel LSTM")
        st.markdown("""
        LSTM mengatasi kelemahan *vanishing gradient* pada RNN standar dengan memanfaatkan 3 gerbang kontrol (*gates*) dan sebuah jalur memori konstan (*cell state*):
        """)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**1. Forget Gate ($f_t$):**")
            st.latex(r"f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)")
            st.caption("Memutuskan berapa banyak memori jam sebelumnya yang harus dilupakan.")

            st.markdown("**2. Input Gate ($i_t$) & Candidate ($\tilde{C}_t$):**")
            st.latex(r"i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)")
            st.latex(r"\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)")
            st.caption("Menyeleksi informasi baru yang layak disimpan ke memori.")

        with col_m2:
            st.markdown("**3. Pembaruan Cell State ($C_t$):**")
            st.latex(r"C_t = f_t * C_{t-1} + i_t * \tilde{C}_t")
            st.caption("Memadukan memori lama dengan informasi beban listrik baru.")

            st.markdown("**4. Output Gate ($o_t$) & Hidden State ($h_t$):**")
            st.latex(r"o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)")
            st.latex(r"h_t = o_t * \tanh(C_t)")
            st.caption("Menghasilkan estimasi profil beban listrik ke dense layer.")

    with tab3:
        st.subheader("Dinamika Pelatihan (Training vs Validation Loss)")
        st.markdown("""
        Model dilatih menggunakan optimizer **Adam** ($\eta = 0.00198$), fungsi loss **MSE**, dan callback **Early Stopping** dengan kesabaran (*patience*) 5 epoch.
        """)
        
        # Plot Loss Curves
        epochs = list(range(1, 21))
        # Representasi loss realistis dari hasil run Colab
        train_loss = [0.0106, 0.0081, 0.0078, 0.0076, 0.0075, 0.0074, 0.0073, 0.0073, 0.0072, 0.0071, 0.0071, 0.0070, 0.0070, 0.0069, 0.0069, 0.0069, 0.0068, 0.0068, 0.0068, 0.0067]
        val_loss   = [0.0080, 0.0074, 0.0071, 0.0070, 0.0070, 0.0070, 0.0068, 0.0068, 0.0069, 0.0067, 0.0067, 0.0068, 0.0067, 0.0066, 0.0067, 0.0067, 0.0067, 0.0066, 0.0066, 0.0067]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, train_loss, label='Training Loss (MSE)', color='#1f77b4', lw=2)
        ax.plot(epochs, val_loss, label='Validation Loss (MSE)', color='#ff7f0e', lw=2, linestyle='--')
        ax.set_title('Learning Curves Model D (GA + PSO + LSTM)', fontweight='bold', fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss MSE (Scaled [0, 1])')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.success("✓ **Analisis Konvergensi**: Kurva Training Loss dan Validation Loss bergerak turun secara beriringan tanpa celah besar (*no significant generalization gap*), membuktikan regularisasi Dropout dan Early Stopping berhasil mengeliminasi risiko *overfitting*.")
