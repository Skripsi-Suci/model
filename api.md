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

---

## 4. Endpoint Pipeline Proses Model (`/api/pipeline/*`)

Endpoint ini merepresentasikan setiap tahapan/proses dalam pengembangan model Random Forest seperti yang tertuang dalam file `Real_Prosessssss.ipynb`.

### A. Load Dataset (`/api/pipeline/load`)
Membaca file dataset `Dataset_coolingwater.csv`, mereset seluruh *state* pipeline yang ada di memory, dan mengembalikan informasi ringkasan dataset.
- **URL:** `/api/pipeline/load`
- **Method:** `POST` atau `GET`
- **Contoh Respons (JSON):**
  ```json
  {
    "status": "success",
    "message": "Dataset loaded successfully.",
    "shape": [1056, 8],
    "columns": ["pH", "SC", "Nitrite", "Fe", "Sulfate", "Turbidity", "Target", "Target_encoded"],
    "class_counts": {
      "Layak": 958,
      "Tidak Layak": 98
    },
    "missing_values": {
      "Fe": 0, "Nitrite": 0, "SC": 0, "Sulfate": 0, "Target": 0, "Target_encoded": 0, "Turbidity": 0, "pH": 0
    }
  }
  ```

### B. Preprocessing Data (`/api/pipeline/preprocess`)
Memeriksa missing values dan data duplikat, lalu menghapus data duplikat dari dataset.
- **URL:** `/api/pipeline/preprocess`
- **Method:** `POST`
- **Contoh Respons (JSON):**
  ```json
  {
    "status": "success",
    "message": "Data preprocessed successfully (duplicates dropped).",
    "duplicate_count": 10,
    "rows_before": 1056,
    "rows_after": 1046,
    "missing_values": {
      "Fe": 0, "Nitrite": 0, "SC": 0, "Sulfate": 0, "Target": 0, "Target_encoded": 0, "Turbidity": 0, "pH": 0
    }
  }
  ```

### C. Train-Test Split (`/api/pipeline/split`)
Membagi data fitur (`X`) dan target (`y`) menjadi data training dan testing (default test size: 30%, stratify=y, random_state=42).
- **URL:** `/api/pipeline/split`
- **Method:** `POST`
- **Body Parameter (JSON - Opsional):**
  - `test_size` (float, default: `0.3`)
  - `random_state` (int, default: `42`)
- **Contoh Respons (JSON):**
  ```json
  {
    "status": "success",
    "message": "Data split successfully.",
    "X_train_shape": [732, 6],
    "X_test_shape": [314, 6],
    "y_train_shape": [732],
    "y_test_shape": [314],
    "features": ["pH", "SC", "Nitrite", "Fe", "Sulfate", "Turbidity"]
  }
  ```

### D. Cross Validation (`/api/pipeline/cv`)
Melakukan K-Fold Cross Validation Stratified sebanyak 5 fold pada data training untuk mengevaluasi model Random Forest awal menggunakan metrik F1-score.
- **URL:** `/api/pipeline/cv`
- **Method:** `POST`
- **Body Parameter (JSON - Opsional):**
  - `n_splits` (int, default: `5`)
  - `random_state` (int, default: `42`)
- **Contoh Respons (JSON):**
  ```json
  {
    "status": "success",
    "fold_f1_scores": [1.0, 0.9962546816479401, 1.0, 1.0, 0.9962264150943396],
    "mean_f1_score": 0.9984962193484559
  }
  ```

### E. Train Model (`/api/pipeline/train`)
Melatih classifier Random Forest pada data training (`X_train`, `y_train`) dan menyimpannya di memory server.
- **URL:** `/api/pipeline/train`
- **Method:** `POST`
- **Body Parameter (JSON - Opsional):**
  - `random_state` (int, default: `42`)
- **Contoh Respons (JSON):**
  ```json
  {
    "status": "success",
    "message": "Model trained successfully."
  }
  ```

### F. Model Evaluation (`/api/pipeline/evaluate`)
Melakukan pengujian model pada data testing (`X_test`) dan mengembalikan akurasi serta rincian Classification Report.
- **URL:** `/api/pipeline/evaluate`
- **Method:** `POST` atau `GET`
- **Contoh Respons (JSON):**
  ```json
  {
    "status": "success",
    "accuracy": 0.9968152866242038,
    "classification_report": {
      "0": {
        "f1-score": 0.9824561403508771,
        "precision": 1.0,
        "recall": 0.9655172413793104,
        "support": 29
      },
      "1": {
        "f1-score": 0.9982486865148862,
        "precision": 0.9965034965034965,
        "recall": 1.0,
        "support": 285
      },
      "accuracy": 0.9968152866242038,
      "macro avg": {
        "f1-score": 0.9903524134328816,
        "precision": 0.9982517482517482,
        "recall": 0.9827586206896552,
        "support": 314
      },
      "weighted avg": {
        "f1-score": 0.9967901392577007,
        "precision": 0.9968264219856576,
        "recall": 0.9968152866242038,
        "support": 314
      }
    }
  }
  ```

### G. Confusion Matrix (`/api/pipeline/confusion-matrix`)
Mengembalikan Confusion Matrix evaluasi test set beserta rincian True Negative (TN), False Positive (FP), False Negative (FN), dan True Positive (TP).
- **URL:** `/api/pipeline/confusion-matrix`
- **Method:** `POST` atau `GET`
- **Contoh Respons (JSON):**
  ```json
  {
    "status": "success",
    "confusion_matrix": [
      [28, 1],
      [0, 285]
    ],
    "tn": 28,
    "fp": 1,
    "fn": 0,
    "tp": 285,
    "classes": ["Tidak Layak", "Layak"]
  }
  ```

### H. Feature Importance (`/api/pipeline/feature-importance`)
Mengambil skor pentingnya fitur (Feature Importance) dari model Random Forest yang telah dilatih, diurutkan dari yang paling penting.
- **URL:** `/api/pipeline/feature-importance`
- **Method:** `POST` atau `GET`
- **Contoh Respons (JSON):**
  ```json
  {
    "status": "success",
    "feature_importance": {
      "Fe": 0.5628439706788642,
      "Turbidity": 0.2326314714765093,
      "pH": 0.06917921889171026,
      "Nitrite": 0.06763505766956049,
      "SC": 0.044389725067828276,
      "Sulfate": 0.023320556215527404
    }
  }
  ```

### I. Save Model (`/api/pipeline/save`)
Menyimpan model Random Forest yang telah dilatih ke dalam file objek binary `.pkl` di server.
- **URL:** `/api/pipeline/save`
- **Method:** `POST`
- **Body Parameter (JSON - Opsional):**
  - `filename` (string, contoh: `"Randomforest_baru.pkl"`)
- **Contoh Respons (JSON):**
  ```json
  {
    "status": "success",
    "message": "Model saved successfully as Randomforest_baru.pkl.",
    "filename": "Randomforest_baru.pkl"
  }
  }
