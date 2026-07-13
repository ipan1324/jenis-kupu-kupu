"""
============================================
predict.py - Modul Prediksi
============================================
Modul untuk memuat model dan melakukan
prediksi klasifikasi jenis kupu-kupu.

Digunakan oleh app.py (Flask) untuk inference.
============================================
"""

import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# ============================================
# Konfigurasi
# ============================================
IMG_SIZE = (299, 299)  # Input size Xception

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "butterfly_xception.keras")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.txt")


def load_labels():
    """
    Memuat daftar label kelas dari file labels.txt.

    Returns:
        list: Daftar nama kelas kupu-kupu

    Raises:
        FileNotFoundError: Jika file labels.txt tidak ditemukan
    """
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            f"File label tidak ditemukan: {LABELS_PATH}. "
            "Jalankan train.py terlebih dahulu."
        )

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    return labels


def load_trained_model():
    """
    Memuat model Xception yang sudah di-training.

    Returns:
        Model: Model Keras yang siap digunakan untuk prediksi

    Raises:
        FileNotFoundError: Jika file model tidak ditemukan
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH}. "
            "Jalankan train.py terlebih dahulu untuk melatih model."
        )

    model = load_model(MODEL_PATH)
    return model


def preprocess_image(image_path):
    """
    Preprocessing gambar untuk prediksi.

    Pipeline:
    1. Buka gambar dengan PIL
    2. Konversi ke RGB (menangani RGBA, grayscale, dsb.)
    3. Resize ke 299x299 (input size Xception)
    4. Konversi ke numpy array
    5. Normalisasi pixel ke [0, 1]
    6. Tambahkan dimensi batch

    Args:
        image_path (str): Path ke file gambar

    Returns:
        np.ndarray: Array gambar siap prediksi (1, 299, 299, 3)

    Raises:
        FileNotFoundError: Jika file gambar tidak ditemukan
        ValueError: Jika file bukan gambar valid
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")

    try:
        # Buka dan konversi gambar
        img = Image.open(image_path)
        img = img.convert("RGB")

        # Resize ke ukuran input Xception
        img = img.resize(IMG_SIZE, Image.LANCZOS)

        # Konversi ke numpy array dan normalisasi
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Tambahkan dimensi batch: (299, 299, 3) -> (1, 299, 299, 3)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        raise ValueError(f"Gagal memproses gambar: {str(e)}")


def predict(image_path):
    """
    Melakukan prediksi klasifikasi kupu-kupu pada gambar.

    Args:
        image_path (str): Path ke file gambar kupu-kupu

    Returns:
        dict: Hasil prediksi berisi:
            - 'class_name' (str): Nama kelas kupu-kupu
            - 'confidence' (float): Tingkat kepercayaan (0-100%)
            - 'all_predictions' (list): Semua probabilitas per kelas
            - 'success' (bool): Status prediksi berhasil
            - 'error' (str): Pesan error jika gagal
    """
    try:
        # Load model dan label
        model = load_trained_model()
        labels = load_labels()

        # Preprocessing gambar
        processed_image = preprocess_image(image_path)

        # Prediksi
        predictions = model.predict(processed_image, verbose=0)

        # Ambil kelas dengan probabilitas tertinggi
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index]) * 100

        # Buat detail prediksi untuk semua kelas
        all_predictions = []
        for i, label in enumerate(labels):
            all_predictions.append({
                "class_name": label,
                "confidence": float(predictions[0][i]) * 100
            })

        # Urutkan berdasarkan confidence tertinggi
        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "success": True,
            "class_name": labels[predicted_index],
            "confidence": round(confidence, 2),
            "all_predictions": all_predictions,
            "error": None
        }

    except FileNotFoundError as e:
        return {
            "success": False,
            "class_name": None,
            "confidence": None,
            "all_predictions": [],
            "error": str(e)
        }

    except Exception as e:
        return {
            "success": False,
            "class_name": None,
            "confidence": None,
            "all_predictions": [],
            "error": f"Terjadi kesalahan saat prediksi: {str(e)}"
        }


# ============================================
# Testing Standalone
# ============================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Penggunaan: python predict.py <path_gambar>")
        print("Contoh: python predict.py static/uploads/test.jpg")
        sys.exit(1)

    image_file = sys.argv[1]
    print(f"\n[INFO] Memproses gambar: {image_file}")

    result = predict(image_file)

    if result["success"]:
        print(f"\n{'=' * 45}")
        print(f" Hasil Prediksi")
        print(f"{'=' * 45}")
        print(f" Kelas   : {result['class_name']}")
        print(f" Confidence: {result['confidence']:.2f}%")
        print(f"\n Detail semua kelas:")
        for pred in result["all_predictions"]:
            bar = "█" * int(pred["confidence"] / 5)
            print(f"  {pred['class_name']:20s} {pred['confidence']:6.2f}% {bar}")
    else:
        print(f"\n[ERROR] {result['error']}")
