# ⚡ Household Electricity Load Forecasting: Hybrid Intelligent System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Institution](https://img.shields.io/badge/ITK-Teknik%20Elektro-green.svg)](https://itk.ac.id/)

Sistem Peramalan Beban Listrik Rumah Tangga Berbasis Hibrida: **Genetic Algorithm (GA)** + **Particle Swarm Optimization (PSO)** + **Long Short-Term Memory (LSTM)** + **Mamdani Fuzzy Inference System (FIS)** terintegrasi dengan **Dashboard Streamlit**.

Proyek Tugas Besar Mata Kuliah **Kecerdasan Buatan (TE2514031)** - Semester Gasal 2026/2027  
**Institut Teknologi Kalimantan (ITK)**  
**Dosen Pengampu**: Ir. Muhammad Ridho Dewanto, S.T., M.T.

---

## 👥 Tim Pengembang (Kelompok 6)

| No | Nama Mahasiswa | NIM | Peran Utama |
|:---:|:---|:---:|:---|
| 1 | **Syahrizal Setiawan Ernaldi** | 04231083 | Lead AI Engineer & GA Feature Selection |
| 2 | **Ferdi Pratama** | 04231032 | Machine Learning Researcher & PSO Hyperparameter Tuning |
| 3 | **Muhammad Umar Alfaradays** | 04241070 | Deep Learning Engineer & LSTM Model Training |
| 4 | **Muhammad Rizky Dimas Pratama** | 04231061 | Decision Support & Streamlit Dashboard Deployment |

---

## 🏗️ Arsitektur Sistem Hibrida

Sistem ini menghubungkan pipeline end-to-end tanpa kebocoran data (*zero data leakage*):

```text
UCI Dataset (2.075.259 baris menit)
       │
       ▼
Preprocessing & Hourly Aggregation (34.565 jam)
       │
       ▼
Chronological Split (70% Train : 15% Val : 15% Test)
       │
       ├─────────────────────────────────┬─────────────────────────────────┐
       ▼                                 ▼                                 ▼
[Genetic Algorithm]            [Particle Swarm Opt.]            [Baseline Model A]
Seleksi Fitur & Lag            Tuning SeqLen & Hyperparams      18 Fitur, W=24h
10 Fitur Terpilih (Pangkas 44%) SeqLen=32h, LR=0.002            Benchmark Acuan
       │                                 │
       └────────────────┬────────────────┘
                        ▼
         [Model D: GA + PSO + LSTM Forecaster]
         MAE Terendah: 0.3169 kW | R²: 0.5887
                        │
                        ▼ (Prediksi Daya Pt+1 & Delta ΔP)
         [Mamdani Fuzzy Inference System (Model E)]
         5 Kategori Status: Very Low, Low, Normal, High, Peak
                        │
                        ▼
         [Streamlit Interactive Cloud Dashboard]
```

---

## 📊 Hasil Eksperimen 4 Model Pembanding (Test Set)

Sesuai ketentuan wajib pengampu, berikut adalah hasil pengujian pada data uji masa depan:

| Model | Konfigurasi | Fitur | Window | Learning Rate | MAE (kW) | RMSE (kW) | R² Score | Waktu Train (s) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | Baseline LSTM | 18 | 24h | 0.0010 | 0.3263 | 0.4646 | 0.5830 | 221.06 |
| **B** | GA + LSTM | 10 | 24h | 0.0010 | 0.3351 | 0.4691 | 0.5750 | 198.98 |
| **C** | PSO + LSTM | 18 | 32h | 0.0020 | 0.3267 | 0.4598 | 0.5919 | 261.59 |
| **D** | **GA + PSO + LSTM** | **10** | **32h** | **0.0020** | **0.3169\*** | **0.4616** | **0.5887** | **245.84** |

*\*Model D mencapai MAE terendah (316,9 Watt) dengan pemangkasan 44,4% variabel input (Pareto-Optimal).*

---

## 🎛️ Sistem Inferensi Fuzzy Mamdani (Model E)

Fuzzy Inference System mengelompokkan hasil peramalan menjadi 5 tingkat kebutuhan daya:
1. **Very Low** (0–20): Konsumsi minimal malam hari / rumah kosong.
2. **Low** (20–40): Beban dasar aktif (kulkas, standby appliances).
3. **Normal** (40–62): Aktivitas wajar harian rata-rata.
4. **High** (62–80): Pemakaian simultan peralatan daya besar.
5. **Peak** (80–100): Kondisi kritis mendekati kapasitas trip MCB daya kontrak.

---

## 💻 Panduan Menjalankan Aplikasi Secara Lokal

1. **Clone Repositori**:
   ```bash
   git clone https://github.com/<username>/household-electricity-forecasting.git
   cd household-electricity-forecasting
   ```

2. **Buat Virtual Environment & Install Dependensi**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Jalankan Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   Aplikasi akan terbuka otomatis di browser pada alamat `http://localhost:8501`.

---

## 🚀 Panduan Deployment ke Streamlit Community Cloud

1. Pastikan seluruh perubahan kode telah di-push ke GitHub:
   ```bash
   git add .
   git commit -m "Siap deploy ke Streamlit Cloud"
   git push origin main
   ```
2. Buka [share.streamlit.io](https://share.streamlit.io/) dan login menggunakan akun GitHub kalian.
3. Klik tombol **New app**.
4. Pilih repositori proyek ini, atur `Branch: main`, dan `Main file path: app.py`.
5. Klik **Deploy!**. Dalam 1–2 menit, dashboard online kalian akan aktif dan dapat diakses publik oleh dosen penguji!
