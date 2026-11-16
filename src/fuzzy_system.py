"""
Modul Fuzzy Inference System (Mamdani) untuk Pendukung Keputusan Beban Listrik (v2)
Kelompok 6 - Household Electricity Load Forecasting
Mata Kuliah: Kecerdasan Buatan (TE2514031) - Institut Teknologi Kalimantan

Ketentuan Dosen & Update v2:
- 5 Kategori Status Beban: Very Low, Low, Normal, High, Peak
- Input 1: Prediksi Daya Aktif P(t+1) dalam kW (Kuantil train: 0.261, 0.402, 1.139, 2.030, 2.942)
- Input 2: Laju Perubahan Beban ΔP = P(t+1) - P(t) dalam kW (Kuantil-70 |dP|: 0.585 kW)
- Output: Indeks Status Beban [0..100], defuzzifikasi Centroid
- 15 Aturan Konsisten (Monoton teruji terhadap P dan ΔP)
"""

import numpy as np


def trimf(x, a, b, c):
    """Triangular membership function: [a, b, c]"""
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)
    if b > a:
        left_mask = (x >= a) & (x <= b)
        y[left_mask] = (x[left_mask] - a) / (b - a)
    if c > b:
        right_mask = (x > b) & (x <= c)
        y[right_mask] = (c - x[right_mask]) / (c - b)
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


def trapmf(x, a, b, c, d):
    """Trapezoidal membership function: [a, b, c, d]"""
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)
    if b > a:
        left_mask = (x >= a) & (x < b)
        y[left_mask] = (x[left_mask] - a) / (b - a)
    plat_mask = (x >= b) & (x <= c)
    y[plat_mask] = 1.0
    if d > c:
        right_mask = (x > c) & (x <= d)
        y[right_mask] = (d - x[right_mask]) / (d - c)
    return np.clip(y, 0.0, 1.0)


def mf_eval(spec, x):
    return trimf(x, *spec[1:]) if spec[0] == 'tri' else trapmf(x, *spec[1:])


class MamdaniFuzzyLoadSystem:
    """
    Sistem Inferensi Fuzzy Mamdani (v2) untuk Penentuan Status Beban Listrik Rumah Tangga.
    """

    STATUS = ['Very Low', 'Low', 'Normal', 'High', 'Peak']
    DELTA = ['Decreasing', 'Steady', 'Increasing']

    STATUS_COLOR = {
        'Very Low': '#10b981',  # Emerald Green
        'Low': '#3b82f6',       # Blue
        'Normal': '#f59e0b',    # Amber
        'High': '#ef4444',      # Red
        'Peak': '#881337'       # Dark Red / Maroon
    }

    RECOMMEND = {
        'Very Low': 'Beban sangat rendah (rumah kosong/malam). Tidak perlu tindakan operasional.',
        'Low': 'Beban dasar aktif (perangkat standby/kulkas). Kondisi kapasitas sistem aman.',
        'Normal': 'Aktivitas rumah tangga normal. Profil beban dalam batas wajar sistem tenaga.',
        'High': 'Pemakaian intensif. Pertimbangkan menggeser operasional alat berdaya besar (mesin cuci/water heater).',
        'Peak': 'PERINGATAN: Beban mendekati kapasitas kontrak daya maksimum / trip MCB. Tunda pemakaian alat berat.'
    }

    RULES_DICT = {
        ('Very Low', 'Decreasing'): 'Very Low',
        ('Very Low', 'Steady'):     'Very Low',
        ('Very Low', 'Increasing'): 'Low',
        ('Low',      'Decreasing'): 'Very Low',
        ('Low',      'Steady'):     'Low',
        ('Low',      'Increasing'): 'Normal',
        ('Normal',   'Decreasing'): 'Low',
        ('Normal',   'Steady'):     'Normal',
        ('Normal',   'Increasing'): 'High',
        ('High',     'Decreasing'): 'Normal',
        ('High',     'Steady'):     'High',
        ('High',     'Increasing'): 'Peak',
        ('Peak',     'Decreasing'): 'High',
        ('Peak',     'Steady'):     'Peak',
        ('Peak',     'Increasing'): 'Peak'
    }

    def __init__(self, quantiles_p=None, delta_scale=0.585, umax=7.0, dmax=5.0):
        # 1. Parameter Kuantil Empiris Dataset Pelatihan (v2)
        if quantiles_p is None:
            c = [0.261, 0.402, 1.139, 2.030, 2.942]
        else:
            c = quantiles_p

        dl = float(delta_scale)
        self.umax = float(umax)
        self.dmax = float(dmax)
        self.quantiles_p = c
        self.delta_scale = dl

        # Semesta Pembicaraan
        self.p_universe = np.linspace(0.0, self.umax, 701)
        self.delta_universe = np.linspace(-self.dmax, self.dmax, 501)
        self.out_universe = np.linspace(0.0, 100.0, 1001)

        # 2. Definisi Parameter Membership Function
        self.p_mf_spec = {
            'Very Low': ['trap', 0.0, 0.0, c[0], c[1]],
            'Low':      ['tri',  c[0], c[1], c[2]],
            'Normal':   ['tri',  c[1], c[2], c[3]],
            'High':     ['tri',  c[2], c[3], c[4]],
            'Peak':     ['trap', c[3], c[4], self.umax, self.umax]
        }

        self.d_mf_spec = {
            'Decreasing': ['trap', -self.dmax, -self.dmax, -dl, 0.0],
            'Steady':     ['tri',  -dl, 0.0, dl],
            'Increasing': ['trap', 0.0, dl, self.dmax, self.dmax]
        }

        self.out_mf_spec = {
            'Very Low': ['trap', 0.0, 0.0, 10.0, 30.0],
            'Low':      ['tri',  10.0, 30.0, 50.0],
            'Normal':   ['tri',  30.0, 50.0, 70.0],
            'High':     ['tri',  50.0, 70.0, 90.0],
            'Peak':     ['trap', 70.0, 90.0, 100.0, 100.0]
        }

        # Lambda accessors untuk backward compatibility
        self.mf_p = {k: (lambda s: lambda x: mf_eval(s, x))(spec) for k, spec in self.p_mf_spec.items()}
        self.mf_delta = {k: (lambda s: lambda x: mf_eval(s, x))(spec) for k, spec in self.d_mf_spec.items()}
        self.mf_out = {k: (lambda s: lambda x: mf_eval(s, x))(spec) for k, spec in self.out_mf_spec.items()}

        self.rules = [(pc, dc, oc) for (pc, dc), oc in self.RULES_DICT.items()]
        self.mu_out = {k: mf_eval(v, self.out_universe) for k, v in self.out_mf_spec.items()}

    def evaluate(self, power_kw: float, delta_kw: float) -> dict:
        """
        Evaluasi inferensi Mamdani lengkap dengan defuzzifikasi Centroid.
        """
        p_val = float(np.clip(power_kw, 0.0, self.umax))
        d_val = float(np.clip(delta_kw, -self.dmax, self.dmax))

        # 1. Fuzzifikasi
        deg_p = {k: float(mf_eval(v, p_val)) for k, v in self.p_mf_spec.items()}
        deg_delta = {k: float(mf_eval(v, d_val)) for k, v in self.d_mf_spec.items()}

        # 2. Implikasi & Agregasi
        aggregated = np.zeros_like(self.out_universe)
        rule_firings = []

        for (pc, dc), oc in self.RULES_DICT.items():
            w = min(deg_p[pc], deg_delta[dc])
            if w > 0.0:
                clipped = np.minimum(w, self.mu_out[oc])
                aggregated = np.maximum(aggregated, clipped)
                rule_firings.append({
                    'rule': f"IF P is {pc} AND dP is {dc} THEN Status is {oc}",
                    'weight': round(float(w), 4)
                })

        # 3. Defuzzifikasi Centroid
        denom = aggregated.sum()
        if denom > 0:
            crisp_score = float((aggregated * self.out_universe).sum() / denom)
        else:
            crisp_score = 50.0

        # 4. Status Linguistik berdasarkan Keanggotaan Terbesar pada Skor Defuzzifikasi
        mu_at_score = [mf_eval(self.out_mf_spec[k], crisp_score) for k in self.STATUS]
        status_idx = int(np.argmax(mu_at_score))
        st_label = self.STATUS[status_idx]

        color_badge = self.STATUS_COLOR[st_label]
        recommendation = self.RECOMMEND[st_label]

        return {
            'power_kw': round(p_val, 4),
            'predicted_power_kw': round(p_val, 4),
            'delta_kw': round(d_val, 4),
            'load_change_delta_kw': round(d_val, 4),
            'fuzzy_score': round(crisp_score, 2),
            'demand_status': st_label,
            'color': color_badge,
            'recommendation': recommendation,
            'active_rules_count': len(rule_firings),
            'active_rules': sorted(rule_firings, key=lambda x: x['weight'], reverse=True),
            'rule_firings': sorted(rule_firings, key=lambda x: x['weight'], reverse=True)[:5],
            'degree_power': {k: round(v, 4) for k, v in deg_p.items()},
            'degree_delta': {k: round(v, 4) for k, v in deg_delta.items()}
        }
