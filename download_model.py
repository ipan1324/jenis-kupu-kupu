"""
============================================
download_model.py - Auto Download Model
============================================
Script untuk mendownload model butterfly_xception.keras
dari Google Drive secara otomatis saat deployment Railway.

Jalankan SEBELUM gunicorn start (lihat Procfile).
============================================
"""

import os
import sys

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "butterfly_xception.keras")

# ============================================================
# File ID model dari Google Drive
# (bisa di-override dengan environment variable MODEL_GDRIVE_FILE_ID)
# ============================================================
DEFAULT_FILE_ID = "1sqmYWmUx1xBN1ECn3fBYW1jXYjAQxhr7"
MODEL_GDRIVE_FILE_ID = os.environ.get("MODEL_GDRIVE_FILE_ID", DEFAULT_FILE_ID)


def download_model():
    """Download model dari Google Drive jika belum ada."""

    # Buat folder model jika belum ada
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Cek apakah model sudah ada
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"[INFO] Model sudah ada ({size_mb:.1f} MB), skip download.")
        return True

    if not MODEL_GDRIVE_FILE_ID:
        print("[ERROR] MODEL_GDRIVE_FILE_ID tidak diset!")
        print("        Set environment variable MODEL_GDRIVE_FILE_ID di Railway.")
        sys.exit(1)

    print(f"[INFO] Model tidak ditemukan. Mendownload dari Google Drive...")
    print(f"[INFO] File ID: {MODEL_GDRIVE_FILE_ID}")

    try:
        import gdown
        url = f"https://drive.google.com/uc?id={MODEL_GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

        if os.path.exists(MODEL_PATH):
            size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            print(f"[OK] Model berhasil didownload ({size_mb:.1f} MB)")
            return True
        else:
            print("[ERROR] Download gagal — file tidak ditemukan setelah download.")
            sys.exit(1)

    except ImportError:
        print("[ERROR] Package 'gdown' tidak terinstall.")
        print("        Tambahkan 'gdown' ke requirements.txt")
        sys.exit(1)

    except Exception as e:
        print(f"[ERROR] Gagal mendownload model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 50)
    print(" Download Model - Butterfly Classification")
    print("=" * 50)
    download_model()
    print("[OK] Siap menjalankan aplikasi.")
