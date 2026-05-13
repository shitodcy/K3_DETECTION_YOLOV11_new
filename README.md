
# Sistem Deteksi Pelanggaran K3 Berbasis Edge Computing

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v11-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-lightgrey)

Proyek ini merupakan implementasi Computer Vision ujung-ke-ujung untuk mendeteksi kepatuhan penggunaan Alat Pelindung Diri pada lingkungan kerja konstruksi dan industri. Sistem ini dirancang secara ringan menggunakan arsitektur **YOLOv11 Nano** agar dapat diimplementasikan pada perangkat Edge Devices seperti NVIDIA Jetson Orin Nano.

---
```markdown
## Fitur Utama

1. **Live Stream:** Inferensi video langsung dari kamera pengawas dengan kalkulasi *Frames Per Second* (FPS) dinamis.
2. **Analisis Gambar Statis:** Deteksi objek pada foto dengan ekstraksi data ke dalam format tabel analitik.
3. **Penyimpanan Bukti Otomatis:** Sistem dibekali logika *auto-save* yang akan secara otomatis mengambil dan menyimpan tangkapan layar jika mendeteksi pekerja tanpa APD lengkap.
4. **Dasboard Interaktif:** Antarmuka pengguna berbasis web modern tanpa memerlukan konfigurasi kode manual saat operasional.
5. **Manajemen Eksperimen (MLOps):** Terintegrasi dengan MLflow dan basis data SQLite untuk pelacakan metrik pelatihan dan manajemen versi model.

**Kategori Deteksi:**
`person` | `helmet` | `vest` | `no-helmet` | `no-vest`

---

## Teknologi yang Digunakan

- **Model Inferensi:** Ultralytics YOLOv11n
- **Antarmuka Web:** Streamlit, Pandas
- **Pemrosesan Gambar:** OpenCV, NumPy, Pillow, Albumentations (Augmentasi)
- **Pelacakan (Tracking):** MLflow, Pyngrok

---

## Persyaratan Sistem

- Sistem Operasi: Linux (Disarankan Arch/EndeavourOS) atau Windows
- Python 3.9 atau lebih baru
- NVIDIA GPU dengan dukungan CUDA (VRAM Minimal 4GB direkomendasikan untuk training)

**Persyaratan Khusus Linux (Arch):**
Pastikan driver CUDA dan cuDNN telah terinstal.
```bash
sudo pacman -S nvidia-dkms nvidia-utils cuda cudnn

```

---

## Instalasi

**1. Kloning Repository**

```bash
git clone [https://github.com/shitodcy/K3_DETECTION_YOLOV11_new/.git]

```

**2. Pembuatan Lingkungan Virtual (Virtual Environment)**

```bash
python -m venv venv
source venv/bin/activate  # Untuk Linux/MacOS
# venv\Scripts\activate   # Untuk Windows

```

**3. Instalasi Dependensi**

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install ultralytics streamlit pandas opencv-python numpy Pillow mlflow pyngrok albumentations

```

---

## Panduan Penggunaan

Sistem ini memiliki dua komponen utama yang dapat dijalankan secara terpisah: **Dasboard Aplikasi** untuk pengguna akhir dan **Server MLflow** untuk pengembang.

### A. Menjalankan Dasboard Utama (Streamlit)

Ini adalah antarmuka operasional yang akan digunakan oleh divisi K3/Keamanan di lapangan.

1. Buka terminal di dalam direktori proyek.
2. Jalankan perintah berikut:

```bash
streamlit run app.py

```

3. Akses dasboard melalui peramban pada alamat lokal yang tertera (umumnya `http://localhost:8501`).

### B. Menjalankan Pelacakan MLflow (Opsional untuk Pengembangan)

Gunakan komponen ini jika Anda ingin melihat grafik performa dari pelatihan model yang telah dilakukan.

1. Pastikan Anda memiliki file `mlflow.db` di direktori proyek.
2. Jalankan server MLflow:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

```

3. Buka `http://localhost:5000` di peramban untuk mengakses grafik akurasi (*Precision, Recall, mAP*) dan artifak model.

---

## Catatan Konfigurasi

* **Penyesuaian Path Model:** Pastikan variabel `MODEL_PATH` di dalam file `app.py` mengarah ke file bobot model terbaik Anda (`best.pt`).
* **Penyesuaian Dataset:** Proyek ini mengambil dataset dari Roboflow 100 (*Construction Safety*). Konfigurasi API *key* dan pra-pemrosesan diatur melalui format buku kerja (`.ipynb`).
```
