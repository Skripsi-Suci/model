---
title: Cooling Water Prediction API
emoji: 💧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Cooling Water Prediction API

REST API prediksi kelayakan air pendingin (*cooling water*) menggunakan
**Random Forest Classifier** + validasi **SOP**. Dibangun dengan Flask dan
dijalankan oleh **Gunicorn** (WSGI server production) di dalam Docker.

## Endpoint

- `GET  /health`  — cek status API.
- `POST /predict` — prediksi kelayakan air (JSON). Lihat [api.md](api.md) untuk detail lengkap.

### Contoh

```bash
curl -X POST https://<username>-<space-name>.hf.space/predict \
     -H "Content-Type: application/json" \
     -d '{"pH": 7.2, "SC": 4900, "Nitrite": 1050, "Fe": 0.8, "Sulfate": 101, "Turbidity": 31}'
```

## Menjalankan secara lokal

Dengan Docker (sama persis seperti di Hugging Face):

```bash
docker build -t cw-api .
docker run -p 7860:7860 cw-api
# API: http://localhost:7860
```

Atau dengan Gunicorn langsung (Linux/Mac):

```bash
pip install -r requirements.txt
gunicorn --bind 0.0.0.0:7860 app:app
```

> Catatan: `python app.py` masih bisa dipakai untuk pengembangan, tapi itu
> development server Flask — **jangan** dipakai untuk production.
