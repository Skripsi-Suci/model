# ============================================================
# model_inference.py
# FUNGSI: Load model dari file .pkl dan prediksi + validasi SOP
# TIDAK ADA TRAINING ULANG
# ============================================================

import pandas as pd
import joblib
import numpy as np

# ============================================================
# 1. LOAD MODEL LANGSUNG (karena disimpan sebagai objek classifier)
# ============================================================
model = joblib.load("Randomforest_cw.pkl")   # pastikan file ini ada
print("Model berhasil dimuat. Siap untuk prediksi.")

# Feature importance diambil dari model yang sudah di-load
feature_importance = model.feature_importances_
feature_names = ['pH', 'SC', 'Nitrite', 'Fe', 'Sulfate', 'Turbidity']
feature_importance_dict = dict(zip(feature_names, feature_importance))

# ============================================================
# 2. STANDAR BATAS PARAMETER (sesuai .ipynb & proposal)
# ============================================================
BATAS_SOP = {
    'pH': (8.0, 11.0),
    'SC': (0.0, 6000.0),
    'Nitrite': (500.0, 1500.0),
    'Fe': (0.0, 1.0),
    'Sulfate': (0.0, 100.0),
    'Turbidity': (0.0, 30.0)
}

# ============================================================
# 3. FUNGSI VALIDASI SOP DAN REKOMENDASI
# ============================================================
def validasi_sop(data, batas_sop=None):
    """Return (status, list_pelanggaran)"""
    limits = batas_sop if batas_sop is not None else BATAS_SOP
    pelanggaran = []
    for param, nilai in data.items():
        if param in limits:
            min_val, max_val = limits[param]
            if not (min_val <= nilai <= max_val):
                pelanggaran.append(f"{param} ({nilai}) di luar batas {min_val}--{max_val}")
    if pelanggaran:
        return "Tidak Layak", pelanggaran
    return "Layak", []

def detail_validasi(data, batas_sop=None):
    """Return dict detail tiap parameter"""
    limits = batas_sop if batas_sop is not None else BATAS_SOP
    hasil = {}
    for param, nilai in data.items():
        if param in limits:
            min_val, max_val = limits[param]
            status = "Normal" if min_val <= nilai <= max_val else "Tidak Normal"
            hasil[param] = {
                "nilai": nilai,
                "min": min_val,
                "max": max_val,
                "status": status
            }
    return hasil

def rekomendasi(data, batas_sop=None):
    limits = batas_sop if batas_sop is not None else BATAS_SOP
    saran = []
    
    # pH
    ph_min, ph_max = limits.get('pH', (8.0, 11.0))
    if data['pH'] < ph_min:
        saran.append(f"pH terlalu rendah ({data['pH']}) → naikkan pH (tambah alkali) [Target SOP: {ph_min} - {ph_max}]")
    elif data['pH'] > ph_max:
        saran.append(f"pH terlalu tinggi ({data['pH']}) → turunkan pH (tambah asam) [Target SOP: {ph_min} - {ph_max}]")
    
    # SC
    sc_min, sc_max = limits.get('SC', (0.0, 6000.0))
    if data['SC'] > sc_max:
        saran.append(f"Konduktivitas tinggi ({data['SC']} µS/cm) → lakukan blowdown [Target SOP: max {sc_max}]")
        
    # Nitrite
    nit_min, nit_max = limits.get('Nitrite', (500.0, 1500.0))
    if data['Nitrite'] < nit_min:
        saran.append(f"Nitrit rendah ({data['Nitrite']} ppm) → tambah inhibitor korosi [Target SOP: {nit_min} - {nit_max}]")
    elif data['Nitrite'] > nit_max:
        saran.append(f"Nitrit tinggi ({data['Nitrite']} ppm) → kurangi dosis inhibitor [Target SOP: {nit_min} - {nit_max}]")
        
    # Fe
    fe_min, fe_max = limits.get('Fe', (0.0, 1.0))
    if data['Fe'] > fe_max:
        saran.append(f"Kandungan Fe tinggi ({data['Fe']} ppm) → indikasi korosi, periksa pipa [Target SOP: max {fe_max}]")
        
    # Sulfate
    sulf_min, sulf_max = limits.get('Sulfate', (0.0, 100.0))
    if data['Sulfate'] > sulf_max:
        saran.append(f"Sulfat tinggi ({data['Sulfate']} ppm) → lakukan blowdown [Target SOP: max {sulf_max}]")
        
    # Turbidity
    turb_min, turb_max = limits.get('Turbidity', (0.0, 30.0))
    if data['Turbidity'] > turb_max:
        saran.append(f"Kekeruhan tinggi ({data['Turbidity']} NTU) → lakukan filtrasi [Target SOP: max {turb_max}]")
        
    if not saran:
        saran.append("Semua parameter dalam kondisi optimal")
    return saran

def warning_system(hasil_sop, label_rf):
    if hasil_sop == "Layak" and label_rf == "Layak":
        return {"level": "INFO", "pesan": "Konsisten: air layak digunakan."}
    elif hasil_sop == "Tidak Layak" and label_rf == "Tidak Layak":
        return {"level": "DANGER", "pesan": "Konsisten: air TIDAK layak. Segera tindakan."}
    elif hasil_sop == "Layak" and label_rf == "Tidak Layak":
        return {"level": "WARNING", "pesan": "Model mendeteksi anomali meski SOP terpenuhi. Pantau ketat."}
    else:  # SOP Tidak Layak, RF Layak
        return {"level": "CRITICAL", "pesan": "KRITIS: SOP melanggar. Utamakan SOP! Jangan gunakan air."}

# ============================================================
# 4. FUNGSI UTAMA PREDIKSI (DIPANGGIL OLEH FLASK)
# ============================================================
def prediksi_air(data_baru, batas_sop=None):
    """
    Parameters:
        data_baru (dict): {'pH': float, 'SC': float, 'Nitrite': float,
                           'Fe': float, 'Sulfate': float, 'Turbidity': float}
        batas_sop (dict, optional): custom limits parsed from frontend
    Returns:
        dict: hasil klasifikasi + validasi SOP + rekomendasi + warning
    """
    # Konversi ke float (antisipasi input string)
    data_baru = {k: float(v) for k, v in data_baru.items()}
    
    # Prediksi dengan Random Forest
    input_df = pd.DataFrame([data_baru])
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    confidence = round(float(max(proba)) * 100, 2)
    label_rf = "Layak" if pred == 1 else "Tidak Layak"
    
    # Validasi SOP
    status_sop, pelanggaran = validasi_sop(data_baru, batas_sop)
    
    return {
        "status_prediksi": label_rf,
        "confidence_score": confidence,
        "validasi_sop": status_sop,
        "pelanggaran": pelanggaran,
        "detail_validasi": detail_validasi(data_baru, batas_sop),
        "rekomendasi": rekomendasi(data_baru, batas_sop),
        "warning": warning_system(status_sop, label_rf),
        "feature_importance": feature_importance_dict   # dari model
    }

# ============================================================
# 5. CONTOH UJI COBA (jika file dijalankan langsung)
# ============================================================
if __name__ == "__main__":
    contoh_data = {
        'pH': 7.2,
        'SC': 4900,
        'Nitrite': 1050,
        'Fe': 0.8,
        'Sulfate': 101,
        'Turbidity': 31
    }
    hasil = prediksi_air(contoh_data)
    print("\n--- HASIL PREDIKSI ---")
    for k, v in hasil.items():
        print(f"{k}: {v}")