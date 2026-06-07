# ============================================================
# model_inference.py
# FUNGSI:
# Load model Random Forest dari file .pkl
# Melakukan prediksi data baru
# Melakukan validasi SOP
# Menentukan status kelayakan final
# ============================================================

import pandas as pd
import joblib

# ============================================================
# 1. LOAD MODEL
# ============================================================

model = joblib.load("Randomforest_cw.pkl")
print("Model berhasil dimuat. Siap untuk prediksi.")

# ============================================================
# 2. FEATURE IMPORTANCE
# ============================================================

feature_names = [
    'pH',
    'SC',
    'Nitrite',
    'Fe',
    'Sulfate',
    'Turbidity'
]

feature_importance = model.feature_importances_

feature_importance_dict = dict(
    zip(feature_names, feature_importance)
)

# ============================================================
# 3. BATAS SOP COOLING WATER
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
# 4. VALIDASI SOP
# ============================================================

def validasi_sop(data):
    """
    Mengembalikan:
    status_sop
    daftar pelanggaran SOP
    """

    pelanggaran = []

    for param, nilai in data.items():

        min_val, max_val = BATAS_SOP[param]

        if not (min_val <= nilai <= max_val):

            pelanggaran.append(
                f"{param} ({nilai}) di luar batas {min_val} - {max_val}"
            )

    if len(pelanggaran) > 0:
        return "Tidak Layak", pelanggaran

    return "Layak", []

# ============================================================
# 5. DETAIL VALIDASI SOP
# ============================================================

def detail_validasi(data):

    hasil = {}

    for param, nilai in data.items():

        min_val, max_val = BATAS_SOP[param]

        hasil[param] = {
            "nilai": nilai,
            "min": min_val,
            "max": max_val,
            "status": (
                "Normal"
                if min_val <= nilai <= max_val
                else "Tidak Normal"
            )
        }

    return hasil

# ============================================================
# 6. FUNGSI PREDIKSI UTAMA
# ============================================================

def prediksi_air(data_baru):
    """
    Parameter:
    data_baru = {
        'pH': float,
        'SC': float,
        'Nitrite': float,
        'Fe': float,
        'Sulfate': float,
        'Turbidity': float
    }

    Return:
    hasil klasifikasi
    """

    # Konversi ke float
    data_baru = {
        k: float(v)
        for k, v in data_baru.items()
    }

    # ========================================================
    # Prediksi Random Forest
    # ========================================================

    input_df = pd.DataFrame([data_baru])

    prediksi = model.predict(input_df)[0]

    probabilitas = model.predict_proba(input_df)[0]

    confidence_score = round(
        max(probabilitas) * 100,
        2
    )

    label_rf = (
        "Layak"
        if prediksi == 1
        else "Tidak Layak"
    )

    # ========================================================
    # Validasi SOP
    # ========================================================

    status_sop, pelanggaran = validasi_sop(data_baru)

    # ========================================================
    # Status Kelayakan Final
    # SOP memiliki prioritas tertinggi
    # ========================================================

    if status_sop == "Tidak Layak":
        status_final = "Tidak Layak"
    else:
        status_final = label_rf

    # ========================================================
    # Return Hasil
    # ========================================================

    return {
        "status_prediksi": label_rf,
        "confidence_score": confidence_score,
        "validasi_sop": status_sop,
        "jumlah_pelanggaran": len(pelanggaran),
        "pelanggaran": pelanggaran,
        "status_final": status_final,
        "detail_validasi": detail_validasi(data_baru),
        "feature_importance": feature_importance_dict
    }

# ============================================================
# 7. CONTOH UJI COBA
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

    print("\n===================================")
    print("HASIL PREDIKSI")
    print("===================================\n")

    print(
        f"Prediksi Random Forest : "
        f"{hasil['status_prediksi']}"
    )

    print(
        f"Confidence Score       : "
        f"{hasil['confidence_score']}%"
    )

    print()

    print(
        f"Validasi SOP           : "
        f"{hasil['validasi_sop']}"
    )

    print()

    print(
        f"Jumlah Pelanggaran SOP : "
        f"{hasil['jumlah_pelanggaran']} Parameter"
    )

    print()

    if hasil["pelanggaran"]:

        print("Parameter Bermasalah:")

        for item in hasil["pelanggaran"]:
            print(f"- {item}")

        print()

    print("Status Kelayakan Final :")
    print(hasil["status_final"].upper())

    print("===================================")