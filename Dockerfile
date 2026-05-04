# Gunakan base image yang ringan (Optimasi LO 5)
FROM python:3.9-slim

# Set folder kerja di dalam container
WORKDIR /app

# Copy daftar dependensi
COPY requirements.txt .

# Install dependensi tanpa menyimpan cache (agar image kecil)
RUN pip install --no-cache-dir -r requirements.txt

# Download data NLTK (Stopwords) agar tidak error saat running
RUN python -m nltk.downloader stopwords

# Copy aset aplikasi yang dibutuhkan saja (Modularitas)
COPY app.py .
COPY pipeline_fake_news.pkl .
COPY templates/ ./templates/

# Jalankan server menggunakan Gunicorn (Standar)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]