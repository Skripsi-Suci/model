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
    'pH': (8, 11),
    'SC': (0, 6000),
    'Nitrite': (500, 1500),
    'Fe': (0, 1),
    'Sulfate': (0, 100),
    'Turbidity': (0, 30)
}

# ============================================================
# 3. FUNGSI VALIDASI SOP DAN REKOMENDASI
# ============================================================
def validasi_sop(data):
    """Return (status, list_pelanggaran)"""
    pelanggaran = []
    for param, nilai in data.items():
        min_val, max_val = BATAS_SOP[param]
        if not (min_val <= nilai <= max_val):
            pelanggaran.append(f"{param} ({nilai}) di luar batas {min_val}--{max_val}")
    if pelanggaran:
        return "Tidak Layak", pelanggaran
    return "Layak", []

def detail_validasi(data):
    """Return dict detail tiap parameter"""
    hasil = {}
    for param, nilai in data.items():
        min_val, max_val = BATAS_SOP[param]
        status = "Normal" if min_val <= nilai <= max_val else "Tidak Normal"
        hasil[param] = {
            "nilai": nilai,
            "min": min_val,
            "max": max_val,
            "status": status
        }
    return hasil

def rekomendasi(data):
    saran = []
    if data['pH'] < 8:
        saran.append("pH terlalu rendah → naikkan pH (tambah alkali)")
    elif data['pH'] > 11:
        saran.append("pH terlalu tinggi → turunkan pH (tambah asam)")
    if data['SC'] > 6000:
        saran.append("Konduktivitas tinggi → lakukan blowdown")
    if data['Nitrite'] < 500:
        saran.append("Nitrit rendah → tambah inhibitor korosi")
    elif data['Nitrite'] > 1500:
        saran.append("Nitrit tinggi → kurangi dosis inhibitor")
    if data['Fe'] > 1:
        saran.append("Kandungan Fe tinggi → indikasi korosi, periksa pipa")
    if data['Sulfate'] > 100:
        saran.append("Sulfat tinggi → lakukan blowdown")
    if data['Turbidity'] > 30:
        saran.append("Kekeruhan tinggi → lakukan filtrasi")
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
def prediksi_air(data_baru):
    """
    Parameters:
        data_baru (dict): {'pH': float, 'SC': float, 'Nitrite': float,
                           'Fe': float, 'Sulfate': float, 'Turbidity': float}
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
    status_sop, pelanggaran = validasi_sop(data_baru)
    
    return {
        "status_prediksi": label_rf,
        "confidence_score": confidence,
        "validasi_sop": status_sop,
        "pelanggaran": pelanggaran,
        "detail_validasi": detail_validasi(data_baru),
        "rekomendasi": rekomendasi(data_baru),
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