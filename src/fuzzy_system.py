"""
Modul Fuzzy Inference System (Mamdani) untuk Pendukung Keputusan Beban Listrik
Kelompok 6 - Household Electricity Load Forecasting
Mata Kuliah: Kecerdasan Buatan (TE2514031) - Institut Teknologi Kalimantan

Ketentuan Dosen:
5 Kategori Status Beban: Very Low, Low, Normal, High, Peak
Input 1: Prediksi Daya Aktif P(t+1) dalam kW
Input 2: Laju Perubahan Beban ΔP = P(t+1) - P(t) dalam kW
"""

import numpy as np


def trimf(x, params):
    """Triangular membership function: params = [a, b, c]"""
    a, b, c = params
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)
    # Left slope
    left_mask = (x >= a) & (x <= b) & (b > a)
    y[left_mask] = (x[left_mask] - a) / (b - a)
    # Right slope
    right_mask = (x > b) & (x <= c) & (c > b)
    y[right_mask] = (c - x[right_mask]) / (c - b)
    # Peak
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


def trapmf(x, params):
    """Trapezoidal membership function: params = [a, b, c, d]"""
    a, b, c, d = params
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)
    # Left slope
    left_mask = (x >= a) & (x < b) & (b > a)
    y[left_mask] = (x[left_mask] - a) / (b - a)
    # Plateau
    plat_mask = (x >= b) & (x <= c)
    y[plat_mask] = 1.0
    # Right slope
    right_mask = (x > c) & (x <= d) & (d > c)
    y[right_mask] = (d - x[right_mask]) / (d - c)
    return np.clip(y, 0.0, 1.0)


class MamdaniFuzzyLoadSystem:
    """
    Sistem Inferensi Fuzzy Mamdani untuk Penentuan Status Beban Listrik Rumah Tangga.
    """

    def __init__(self):
        # 1. Semesta Pembicaraan (Universe of Discourse)
        self.p_universe = np.linspace(0.0, 7.0, 701)       # Daya Aktif [0..7 kW]
        self.delta_universe = np.linspace(-2.0, 2.0, 401)   # Perubahan Daya [-2..+2 kW]
        self.out_universe = np.linspace(0.0, 100.0, 1001)   # Indeks Beban [0..100]

        # 2. Parameter Fungsi Keanggotaan Input 1: Daya Aktif P (kW)
        # Berbasis kuantil empiris dataset UCI
        self.mf_p = {
            'Very Low': lambda x: trapmf(x, [0.0, 0.0, 0.25, 0.45]),
            'Low':      lambda x: trimf(x, [0.30, 0.60, 0.95]),
            'Normal':   lambda x: trimf(x, [0.80, 1.25, 1.80]),
            'High':     lambda x: trimf(x, [1.60, 2.30, 3.20]),
            'Peak':     lambda x: trapmf(x, [2.80, 3.80, 7.00, 7.00])
        }

        # 3. Parameter Fungsi Keanggotaan Input 2: Laju Perubahan ΔP (kW)
        self.mf_delta = {
            'Decreasing': lambda x: trapmf(x, [-2.0, -2.0, -0.60, -0.10]),
            'Steady':     lambda x: trimf(x, [-0.25, 0.0, 0.25]),
            'Increasing': lambda x: trapmf(x, [0.10, 0.60, 2.0, 2.0])
        }

        # 4. Parameter Fungsi Keanggotaan Output: Status Beban [0..100]
        self.mf_out = {
            'Very Low': lambda x: trapmf(x, [0.0, 0.0, 12.0, 25.0]),
            'Low':      lambda x: trimf(x, [18.0, 32.0, 48.0]),
            'Normal':   lambda x: trimf(x, [40.0, 52.0, 68.0]),
            'High':     lambda x: trimf(x, [60.0, 74.0, 88.0]),
            'Peak':     lambda x: trapmf(x, [80.0, 92.0, 100.0, 100.0])
        }

        # 5. Rule Base IF-THEN (15 Aturan Konsisten)
        self.rules = [
            ('Very Low', 'Decreasing', 'Very Low'),
            ('Very Low', 'Steady',     'Very Low'),
            ('Very Low', 'Increasing', 'Low'),
            ('Low',      'Decreasing', 'Very Low'),
            ('Low',      'Steady',     'Low'),
            ('Low',      'Increasing', 'Normal'),
            ('Normal',   'Decreasing', 'Low'),
            ('Normal',   'Steady',     'Normal'),
            ('Normal',   'Increasing', 'High'),
            ('High',     'Decreasing', 'Normal'),
            ('High',     'Steady',     'High'),
            ('High',     'Increasing', 'Peak'),
            ('Peak',     'Decreasing', 'High'),
            ('Peak',     'Steady',     'Peak'),
            ('Peak',     'Increasing', 'Peak'),
        ]

    def evaluate(self, power_kw: float, delta_kw: float) -> dict:
        """
        Melakukan evaluasi inferensi Mamdani lengkap:
        Fuzzifikasi -> Implikasi Min -> Agregasi Max -> Defuzzifikasi Centroid
        """
        power_kw = float(np.clip(power_kw, 0.0, 7.0))
        delta_kw = float(np.clip(delta_kw, -2.0, 2.0))

        # 1. Fuzzifikasi Input
        deg_p = {cat: float(func(np.array([power_kw]))[0]) for cat, func in self.mf_p.items()}
        deg_delta = {cat: float(func(np.array([delta_kw]))[0]) for cat, func in self.mf_delta.items()}

        # 2. Evaluasi Rule Base & Implikasi Min
        # Inisialisasi area agregasi output
        aggregated_output = np.zeros_like(self.out_universe)
        rule_firings = []

        for p_cat, delta_cat, out_cat in self.rules:
            # Operator AND (T-Norm: Minimum)
            weight = min(deg_p[p_cat], deg_delta[delta_cat])
            if weight > 0.0:
                # Implikasi Min
                rule_out_mf = self.mf_out[out_cat](self.out_universe)
                rule_clipped = np.minimum(weight, rule_out_mf)
                # Agregasi Max (S-Norm: Maximum)
                aggregated_output = np.maximum(aggregated_output, rule_clipped)
                rule_firings.append({
                    'rule': f"IF Power is {p_cat} AND Delta is {delta_cat} THEN Status is {out_cat}",
                    'weight': round(weight, 4)
                })

        # 3. Defuzzifikasi Centroid (Center of Gravity)
        denom = np.sum(aggregated_output)
        if denom == 0.0:
            # Fallback jika tidak ada rule aktif (default ke batas kuantil P)
            crisp_score = 50.0
        else:
            crisp_score = float(np.sum(self.out_universe * aggregated_output) / denom)

        # 4. Penentuan Kategori Linguistik Akhir
        if crisp_score < 20.0:
            linguistic_status = 'Very Low'
            action_recommendation = 'Konsumsi listrik sangat minimal (kondisi malam / rumah kosong). Efisiensi optimal.'
            color_badge = '#2ca02c'  # Hijau
        elif crisp_score < 40.0:
            linguistic_status = 'Low'
            action_recommendation = 'Beban dasar aktif (kulkas, standby appliances). Kapasitas daya aman.'
            color_badge = '#1f77b4'  # Biru
        elif crisp_score < 62.0:
            linguistic_status = 'Normal'
            action_recommendation = 'Aktivitas normal rumah tangga. Profil beban dalam batas wajar sistem tenaga.'
            color_badge = '#ff7f0e'  # Oranye
        elif crisp_score < 80.0:
            linguistic_status = 'High'
            action_recommendation = 'Penggunaan alat listrik intensif (AC, mesin cuci, microwave bersamaan). Perlu pemantauan.'
            color_badge = '#d62728'  # Merah
        else:
            linguistic_status = 'Peak'
            action_recommendation = 'PERINGATAN: Beban mendekati kapasitas kontrak daya maksimum / batas trip MCB. Dianjurkan load shedding / mematikan alat berdaya besar.'
            color_badge = '#7f1416'  # Merah Tua / Maroon

        return {
            'power_kw': round(power_kw, 4),
            'predicted_power_kw': round(power_kw, 4),
            'delta_kw': round(delta_kw, 4),
            'load_change_delta_kw': round(delta_kw, 4),
            'fuzzy_score': round(crisp_score, 2),
            'demand_status': linguistic_status,
            'color': color_badge,
            'recommendation': action_recommendation,
            'active_rules_count': len(rule_firings),
            'active_rules': sorted(rule_firings, key=lambda x: x['weight'], reverse=True),
            'rule_firings': sorted(rule_firings, key=lambda x: x['weight'], reverse=True)[:5],
            'degree_power': {k: round(v, 4) for k, v in deg_p.items()},
            'degree_delta': {k: round(v, 4) for k, v in deg_delta.items()}
        }
