# 📊 STOCK PREDICTOR IDX

Deskripsi singkat tentang project ini. Jelaskan tujuan, fitur utama, atau problem yang ingin diselesaikan.

---

## 🚀 Cara Menjalankan Project

### 1. Clone Repository

```bash
git clone https://github.com/username/namaproject.git
cd namaproject
```

### 2. Buat Virtual Environment

Supaya dependency rapi & tidak bentrok dengan sistem.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Pastikan ada file `requirements.txt` di repo. Kalau belum, bisa generate dari environment lama:

```bash
pip freeze > requirements.txt
```

Di device baru cukup jalankan:

```bash
pip install -r requirements.txt
```

📦 Minimal dependency yang dibutuhkan:

* streamlit
* plotly
* pandas
* numpy
* scikit-learn
* joblib
* yfinance
* ta
* pyarrow

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

---

## 📂 Struktur Folder (Opsional)

```bash
namaproject/
│
├── src/         # data.py, features.py, label.py, model.py
├── models/      # model hasil training (.joblib)
├── data/        # cache atau file parquet
├── app.py       # main app streamlit
└── requirements.txt
```

---

## ☁️ Deployment

Jika ingin sharing aplikasi:

* **Streamlit Cloud** → Gratis & mudah, cukup hubungkan repo GitHub.
* **Render / VPS** → Lebih bebas & fleksibel.

---

## 📝 Lisensi

Tentukan lisensi project Anda, misalnya MIT, Apache 2.0, atau GPL.

---

💡 **Tips:** Update `README.md` ini sesuai kebutuhan project, tambahkan screenshot aplikasi agar lebih menarik!
