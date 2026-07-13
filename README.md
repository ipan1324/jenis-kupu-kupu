# 🦋 Klasifikasi Jenis Kupu-kupu dengan Transfer Learning Xception

> **Penerapan Transfer Learning Xception untuk Klasifikasi Jenis Kupu-kupu Berbasis Web Menggunakan Flask**

Proyek ini merupakan aplikasi web untuk klasifikasi jenis kupu-kupu menggunakan **Transfer Learning** dengan arsitektur **Xception** yang di-pretrain pada **ImageNet**. Aplikasi dibangun menggunakan **Flask** sebagai backend dan **Bootstrap 5** sebagai frontend.

**Tugas Mata Kuliah:** Artificial Intelligence — Bab 12 Transfer Learning

---

## 📋 Daftar Isi

- [Deskripsi](#-deskripsi)
- [Fitur](#-fitur)
- [Spesies yang Didukung](#-spesies-yang-didukung)
- [Teknologi](#-teknologi)
- [Arsitektur Model](#-arsitektur-model)
- [Struktur Folder](#-struktur-folder)
- [Instalasi](#-instalasi)
- [Persiapan Dataset](#-persiapan-dataset)
- [Training Model](#-training-model)
- [Menjalankan Aplikasi](#-menjalankan-aplikasi)
- [Screenshot](#-screenshot)
- [Deploy ke Render](#-deploy-ke-render)

---

## 📖 Deskripsi

Aplikasi ini menggunakan **Transfer Learning** dengan model **Xception** untuk mengklasifikasikan 5 jenis kupu-kupu dari gambar yang diupload oleh pengguna. Model Xception yang telah di-pretrain pada dataset ImageNet digunakan sebagai feature extractor, kemudian ditambahkan custom classification head untuk mengenali jenis kupu-kupu.

### Mengapa Xception?

- Arsitektur **depthwise separable convolution** yang efisien
- Performa tinggi pada ImageNet (top-1 accuracy 79%)
- Cocok untuk **Transfer Learning** karena fitur yang kaya
- Input size 299×299 memberikan detail yang cukup

---

## ✨ Fitur

- 🔍 **Klasifikasi otomatis** jenis kupu-kupu dari gambar
- 📊 **Distribusi probabilitas** untuk semua kelas
- 🖼️ **Preview gambar** sebelum prediksi
- 🎯 **Drag & Drop** upload gambar
- 📱 **Responsive design** (mobile-friendly)
- 🎨 **UI Modern** dengan tema kupu-kupu biru-ungu
- ⚡ **Real-time prediction** tanpa reload halaman

---

## 🦋 Spesies yang Didukung

| No | Nama | Nama Latin |
|----|------|------------|
| 1 | **Monarch** | *Danaus plexippus* |
| 2 | **Peacock** | *Aglais io* |
| 3 | **Julia** | *Dryas iulia* |
| 4 | **Viceroy** | *Limenitis archippus* |
| 5 | **Zebra Long Wing** | *Heliconius charithonia* |

---

## 🛠️ Teknologi

| Komponen | Teknologi |
|----------|-----------|
| Bahasa | Python 3.11 |
| Deep Learning | TensorFlow 2.x, Keras |
| Arsitektur | Xception (Transfer Learning) |
| Web Framework | Flask |
| Frontend | Bootstrap 5, HTML5, CSS3 |
| Image Processing | Pillow, NumPy |
| Dataset | Kaggle - Butterfly Image Classification |

---

## 🏗️ Arsitektur Model

```
Input (299, 299, 3)
        │
   ┌────▼─────┐
   │  Xception │  ← Pretrained ImageNet (Frozen)
   │ Base Model│
   └────┬──────┘
        │
  GlobalAveragePooling2D
        │
    Dropout(0.5)
        │
  Dense(256, ReLU)
        │
    Dropout(0.3)
        │
  Dense(5, Softmax) → Output (5 kelas)
```

**Konfigurasi Training:**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 15 (dengan EarlyStopping)
- Batch Size: 16
- Callbacks: EarlyStopping (patience=5), ModelCheckpoint

---

## 📁 Struktur Folder

```
butterfly-classification/
│
├── dataset/                    # Dataset kupu-kupu
│   ├── train/                  # Data training
│   │   ├── Julia/
│   │   ├── Monarch/
│   │   ├── Peacock/
│   │   ├── Viceroy/
│   │   └── Zebra Long Wing/
│   ├── validation/             # Data validasi
│   │   └── (sama seperti train)
│   └── test/                   # Data testing
│       └── (sama seperti train)
│
├── model/                      # Model tersimpan
│   ├── butterfly_xception.keras
│   └── labels.txt
│
├── static/                     # File statis
│   ├── uploads/                # Gambar yang diupload
│   └── css/
│       └── style.css           # Custom stylesheet
│
├── templates/                  # Template HTML
│   ├── index.html              # Halaman utama
│   └── result.html             # Halaman hasil prediksi
│
├── app.py                      # Aplikasi Flask (Controller)
├── train.py                    # Script training model
├── predict.py                  # Modul prediksi (Model)
├── requirements.txt            # Dependencies Python
├── README.md                   # Dokumentasi
└── .gitignore                  # Git ignore
```

---

## ⚙️ Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/ipan1324/jenis-kupu-kupu.git
cd jenis-kupu-kupu
```

### 2. Buat Virtual Environment

```bash
python -m venv venv
```

**Aktifkan virtual environment:**

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Persiapan Dataset

### ☁️ Download Dataset (Google Drive)

Dataset telah tersedia di Google Drive dan dapat diunduh langsung:

> **[📥 Download Dataset Kupu-kupu — Google Drive](https://drive.google.com/drive/folders/1ZlPvmMqYRTwMPEXGyy0Oug5b26vohowz?usp=sharing)**

Dataset mencakup 5 kelas kupu-kupu:
- `Julia`
- `Monarch`
- `Peacock`
- `Viceroy`
- `Zebra Long Wing`

### Struktur Dataset

Setelah download, susun dataset ke dalam struktur berikut:

```
dataset/
├── train/
│   ├── Julia/          # ~100+ gambar
│   ├── Monarch/        # ~100+ gambar
│   ├── Peacock/        # ~100+ gambar
│   ├── Viceroy/        # ~100+ gambar
│   └── Zebra Long Wing/  # ~100+ gambar
├── validation/
│   ├── Julia/
│   ├── Monarch/
│   ├── Peacock/
│   ├── Viceroy/
│   └── Zebra Long Wing/
└── test/
    ├── Julia/
    ├── Monarch/
    ├── Peacock/
    ├── Viceroy/
    └── Zebra Long Wing/
```

> **Tip:** Jika dataset hanya memiliki folder `train` dan `test`, Anda bisa membagi folder `train` secara manual (80% train, 20% validation).

> **Catatan:** Folder `archive/` dan `dataset/` tidak disertakan di repository ini karena ukurannya yang besar. Gunakan link Google Drive di atas.

---

## 🚀 Training Model

Setelah dataset siap, jalankan training:

```bash
python train.py
```

**Output training:**
- Model terbaik disimpan di: `model/butterfly_xception.keras`
- Label kelas disimpan di: `model/labels.txt`
- Training menggunakan EarlyStopping (berhenti otomatis jika tidak ada peningkatan)

**Estimasi waktu:**
- GPU: ~5-10 menit
- CPU: ~30-60 menit

---

## 🌐 Menjalankan Aplikasi

Setelah model selesai di-training:

```bash
python app.py
```

Buka browser dan akses:

```
http://127.0.0.1:5000
```

### Cara Penggunaan:
1. Buka halaman utama
2. Upload gambar kupu-kupu (JPG/JPEG/PNG, maks 10 MB)
3. Klik tombol **"Prediksi Sekarang"**
4. Lihat hasil klasifikasi dan confidence score
5. Klik **"Upload Gambar Lain"** untuk prediksi baru

---

## 📸 Screenshot

### Halaman Utama
> *[Screenshot halaman utama dengan form upload]*

### Hasil Prediksi
> *[Screenshot hasil prediksi dengan confidence score]*

---

## ☁️ Deploy ke Render

### 1. Persiapan

Buat file `Procfile` di root project:

```
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

Buat file `runtime.txt`:

```
python-3.11.9
```

### 2. Push ke GitHub

```bash
git init
git add .
git commit -m "Initial commit - Butterfly Classification"
git remote add origin https://github.com/ipan1324/jenis-kupu-kupu.git
git push -u origin main
```

### 3. Deploy di Render

1. Buka [render.com](https://render.com)
2. Klik **"New +"** → **"Web Service"**
3. Hubungkan repository GitHub
4. Konfigurasi:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Instance Type:** Free atau Starter
5. Klik **"Create Web Service"**

> **Catatan:** File model (`butterfly_xception.keras`) tidak disertakan di repository ini karena ukurannya yang besar (±86 MB). Lakukan training ulang menggunakan `python train.py` setelah dataset didownload, atau minta akses model secara terpisah.

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik (tugas mata kuliah AI).

Dataset: [Butterfly Image Classification - Kaggle](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)

---

## 👨‍💻 Pengembang

Dibuat sebagai tugas **Mata Kuliah Artificial Intelligence** — **Bab 12: Transfer Learning Xception**

---

<p align="center">
  🦋 <em>Butterfly Classification — Transfer Learning Xception + Flask</em> 🦋
</p>
