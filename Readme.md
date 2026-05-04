# 🛡️ Fake News Detection System: End-to-End MLOps Pipeline

[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Proyek ini adalah sistem deteksi berita palsu (*Fake News*) yang dibangun dengan pendekatan **MLOps** (Machine Learning Operations). Sistem ini mencakup seluruh siklus hidup pengembangan, mulai dari riset data, pelatihan model, hingga pengembangan Web Service yang dikontainerisasi menggunakan Docker.

---

## 🏗️ Software Architecture (LO 3)
Proyek ini mengadopsi arsitektur yang kohesif dengan pemisahan tanggung jawab (*Separation of Concerns*) yang jelas:

1.  **Research Layer (`/Jupyter`)**: Berisi eksperimen data, pra-pemrosesan NLP, dan pelatihan model.
2.  **Logic Layer (`app.py`)**: Web service berbasis Flask yang menangani permintaan inferensi secara real-time.
3.  **UI Layer (`/templates`)**: Antarmuka pengguna yang responsif menggunakan Tailwind CSS.
4.  **Infrastruktur Layer (`Dockerfile` & `docker-compose.yml`)**: Abstraksi lingkungan menggunakan Docker untuk memastikan skalabilitas dan portabilitas.

---

## 🔬 Modeling & Algorithm (LO 4)
Model dilatih menggunakan **ISOT Dataset** (internasional) dengan alur kerja sebagai berikut:

### 1. Preprocessing (NLP)
Teks dibersihkan melalui fungsi `clean_text` yang mencakup:
* **Case Folding**: Mengubah teks ke lowercase.
* **Noise Removal**: Menghapus angka, tanda baca, dan teks dalam kurung.
* **Stopwords Removal**: Menggunakan `nltk.corpus.stopwords` bahasa Inggris untuk membuang kata-kata umum yang tidak bermakna.

### 2. Feature Extraction
Menggunakan **TF-IDF Vectorizer** untuk mengubah teks menjadi representasi vektor numerik. Algoritma ini memberikan bobot tinggi pada kata-kata unik yang menjadi ciri khas berita palsu vs asli.

### 3. Classification Model
Menggunakan **Logistic Regression** dengan fungsi aktivasi Sigmoid:
$$P(y=1|x) = \frac{1}{1 + e^{-z}}$$
Model ini menghasilkan akurasi sebesar **98.58%** pada data pengujian.

---

## 🐳 Web Service Deployment (LO 5)
Aplikasi ini sepenuhnya dikontainerisasi menggunakan **Docker** untuk menjamin lingkungan yang *reproducible* dan *optimized*.

### Mengapa Docker?
* **Isolasi Environment**: Mengunci versi Python (3.9-slim) dan library agar identik di semua komputer.
* **Production Ready**: Menggunakan **Gunicorn** sebagai WSGI HTTP Server untuk stabilitas tinggi.
* **Automated Setup**: Docker secara otomatis mengunduh dependensi NLTK saat proses *build*.

### Struktur Folder
```text
AOL_MACHINELEARNING/
├── Jupyter/                    # Area Riset (Eksperimen)
│   ├── AOL_ML_FakeNews_Research.ipynb
│   ├── Fake.csv
│   └── True.csv
├── templates/                  # Frontend UI
│   └── index.html
├── app.py                      # Flask API & Logic
├── Dockerfile                  # Blueprint Container
├── docker-compose.yml          # Orkestrasi Service
├── pipeline_fake_news.pkl      # Saved Model (Pickle)
├── requirements.txt            # Daftar Dependensi
└── README.md                   # Dokumentasi Teknis