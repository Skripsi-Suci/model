# Dokumentasi API Prediksi Kelayakan Air Cooling Water

API ini digunakan untuk memprediksi kelayakan air pendingin (*cooling water*) menggunakan model **Random Forest Classifier** serta memvalidasinya berdasarkan standar **Batas SOP (Standard Operating Procedure)** yang telah ditentukan.

API ini dibangun menggunakan framework **Flask** di Python.

---

## 1. Cara Menjalankan API Server

Sebelum menggunakan API, pastikan Anda telah menginstal dependensi yang diperlukan (seperti `Flask`, `pandas`, `joblib`, `scikit-learn`, dll.) dan file model `Randomforest_cw.pkl` berada di direktori yang sama dengan `app.py` dan `model.py`.

Jalankan server menggunakan perintah berikut di terminal:
```bash
python app.py
```
Secara default, server akan berjalan pada alamat: `http://localhost:5000` atau `http://127.0.0.1:5000`

---

## 2. Endpoint Prediksi (`/predict`)

Menerima data parameter kualitas air dalam format JSON, melakukan klasifikasi kelayakan menggunakan model Random Forest, mencocokkan nilai dengan batas SOP, dan memberikan rekomendasi serta level peringatan (*warning system*).

- **URL:** `/predict`
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`

### A. Parameter Request (JSON)

Semua parameter berikut berstatus **wajib (required)** dan bertipe data angka (`float` atau `int`):

| Parameter | Deskripsi | Standar Batas SOP |
| :--- | :--- | :--- |
| `pH` | Tingkat keasaman air | `8` s/d `11` |
| `SC` | *Specific Conductance* (Konduktivitas) | `0` s/d `6000` |
| `Nitrite` | Kandungan Nitrit | `500` s/d `1500` |
| `Fe` | Kandungan Zat Besi | `0` s/d `1` |
| `Sulfate` | Kandungan Sulfat | `0` s/d `100` |
| `Turbidity` | Kekeruhan air | `0` s/d `30` |

#### Contoh Request Body (JSON)
```json
{
  "pH": 7.2,
  "SC": 4900,
  "Nitrite": 1050,
  "Fe": 0.8,
  "Sulfate": 101,
  "Turbidity": 31
}
```

---

### B. Cara Pengujian Endpoint

#### Menggunakan `cURL` (Command Line)
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d "{\"pH\": 7.2, \"SC\": 4900, \"Nitrite\": 1050, \"Fe\": 0.8, \"Sulfate\": 101, \"Turbidity\": 31}"
```

#### Menggunakan Python (`requests`)
```python
import requests

url = "http://localhost:5000/predict"
data = {
    "pH": 7.2,
    "SC": 4900,
    "Nitrite": 1050,
    "Fe": 0.8,
    "Sulfate": 101,
    "Turbidity": 31
}

response = requests.post(url, json=data)
print(response.json())
```

---

### C. Respons API (Responses)

#### 1. Respons Berhasil (HTTP Status `200 OK`)
Dikembalikan jika seluruh parameter valid dan berhasil diproses oleh model.

```json
{
  "status_prediksi": "Tidak Layak",
  "confidence_score": 98.2,
  "validasi_sop": "Tidak Layak",
  "pelanggaran": [
    "pH (7.2) di luar batas 8--11",
    "Sulfate (101.0) di luar batas 0--100",
    "Turbidity (31.0) di luar batas 0--30"
  ],
  "detail_validasi": {
    "Fe": {
      "max": 1,
      "min": 0,
      "nilai": 0.8,
      "status": "Normal"
    },
    "Nitrite": {
      "max": 1500,
      "min": 500,
      "nilai": 1050.0,
      "status": "Normal"
    },
    "SC": {
      "max": 6000,
      "min": 0,
      "nilai": 4900.0,
      "status": "Normal"
    },
    "Sulfate": {
      "max": 100,
      "min": 0,
      "nilai": 101.0,
      "status": "Tidak Normal"
    },
    "Turbidity": {
      "max": 30,
      "min": 0,
      "nilai": 31.0,
      "status": "Tidak Normal"
    },
    "pH": {
      "max": 11,
      "min": 8,
      "nilai": 7.2,
      "status": "Tidak Normal"
    }
  },
  "rekomendasi": [
    "pH terlalu rendah → naikkan pH (tambah alkali)",
    "Sulfat tinggi → lakukan blowdown",
    "Kekeruhan tinggi → lakukan filtrasi"
  ],
  "warning": {
    "level": "DANGER",
    "pesan": "Konsisten: air TIDAK layak. Segera tindakan."
  },
  "feature_importance": {
    "Fe": 0.15829103,
    "Nitrite": 0.12450291,
    "SC": 0.11985721,
    "Sulfate": 0.20183742,
    "Turbidity": 0.21849104,
    "pH": 0.17702042
  }
}
```

##### Penjelasan Field Respons:
- **`status_prediksi`**: Hasil klasifikasi model Machine Learning (Random Forest). Nilainya `Layak` atau `Tidak Layak`.
- **`confidence_score`**: Tingkat keyakinan model Random Forest dalam melakukan prediksi (persentase `0` - `100`).
- **`validasi_sop`**: Hasil pencocokan parameter secara langsung dengan aturan SOP. Nilainya `Layak` (jika semua parameter normal) atau `Tidak Layak` (jika ada minimal satu parameter abnormal).
- **`pelanggaran`**: List berisi parameter apa saja yang nilainya di luar rentang SOP beserta keterangannya.
- **`detail_validasi`**: Penjelasan mendetail untuk setiap parameter yang dikirimkan, lengkap dengan nilai input, nilai batas min/max, dan status kelayakannya (`Normal` atau `Tidak Normal`).
- **`rekomendasi`**: Instruksi tindakan yang disarankan untuk memperbaiki kualitas air berdasarkan parameter yang melanggar standar SOP.
- **`warning`**: Sistem peringatan gabungan dari validasi SOP dan prediksi model Random Forest.
- **`feature_importance`**: Tingkat pengaruh dari masing-masing parameter terhadap hasil keputusan model Random Forest.

#### 2. Respons Bad Request (HTTP Status `400 Bad Request`)
Dikembalikan jika ada satu atau beberapa parameter wajib yang tidak disertakan dalam request body.

```json
{
  "error": "Missing parameters. Required: pH, SC, Nitrite, Fe, Sulfate, Turbidity"
}
```

#### 3. Respons Server Error (HTTP Status `500 Internal Server Error`)
Dikembalikan jika terjadi kesalahan internal di sisi server (misal, tipe data salah/tidak bisa dikonversi ke angka).

```json
{
  "error": "could not convert string to float: 'abc'"
}
```

---

## 3. Matriks Logika Peringatan (*Warning System*)

Sistem memberikan level peringatan berdasarkan kecocokan antara **Validasi SOP** dan **Prediksi Random Forest**:

| Validasi SOP | Prediksi Random Forest | Level Peringatan | Pesan / Deskripsi |
| :--- | :--- | :--- | :--- |
| **Layak** | **Layak** | `INFO` | Konsisten: air layak digunakan. |
| **Tidak Layak** | **Tidak Layak** | `DANGER` | Konsisten: air TIDAK layak. Segera tindakan. |
| **Layak** | **Tidak Layak** | `WARNING` | Model mendeteksi anomali meski SOP terpenuhi. Pantau ketat. |
| **Tidak Layak** | **Layak** | `CRITICAL` | KRITIS: SOP melanggar. Utamakan SOP! Jangan gunakan air. |
