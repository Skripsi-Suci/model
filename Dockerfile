# Dockerfile untuk Hugging Face Spaces (SDK: docker)
# Menjalankan Flask API dengan Gunicorn (WSGI server production).

FROM python:3.13-slim

# Hugging Face menjalankan container sebagai user id 1000 (best practice)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependency lebih dulu agar layer cache efisien
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Salin sisa kode + file model .pkl
COPY --chown=user . .

# Hugging Face Spaces (Docker) memakai port 7860 secara default
EXPOSE 7860

# Production WSGI server. "app:app" = file app.py, objek Flask bernama `app`.
#  --workers 2     : 2 proses worker (cocok untuk CPU basic 2 vCPU)
#  --threads 4     : tiap worker melayani beberapa request sekaligus
#  --timeout 120   : beri waktu lebih untuk load model di awal
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]
