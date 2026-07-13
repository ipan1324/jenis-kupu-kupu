"""
============================================
app.py - Aplikasi Web Flask
============================================
Aplikasi web untuk klasifikasi jenis kupu-kupu
menggunakan Transfer Learning Xception.

Arsitektur MVC sederhana:
- Model: predict.py (logika AI)
- View: templates/ (HTML + Bootstrap 5)
- Controller: app.py (routing Flask)
============================================
"""

import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from predict import predict


# ============================================
# Konfigurasi Aplikasi
# ============================================
app = Flask(__name__)
app.secret_key = "butterfly-classification-secret-key-2024"

# Konfigurasi upload
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # Maksimal 10 MB

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Buat folder uploads jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """
    Validasi ekstensi file yang diizinkan.

    Hanya menerima file JPG, JPEG, dan PNG.

    Args:
        filename (str): Nama file yang diupload

    Returns:
        bool: True jika ekstensi diizinkan
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ============================================
# Routes
# ============================================

@app.route("/", methods=["GET"])
def index():
    """
    Halaman utama - Form upload gambar kupu-kupu.
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_image():
    """
    Route untuk memproses upload gambar dan melakukan prediksi.

    Validasi:
    1. Cek apakah file ada dalam request
    2. Cek apakah file dipilih
    3. Cek ekstensi file (JPG, JPEG, PNG)

    Jika valid, simpan file dan jalankan prediksi.
    """
    # Validasi: cek apakah file ada dalam request
    if "file" not in request.files:
        flash("Tidak ada file yang dipilih. Silakan pilih gambar.", "danger")
        return redirect(url_for("index"))

    file = request.files["file"]

    # Validasi: cek apakah file dipilih
    if file.filename == "":
        flash("Tidak ada file yang dipilih. Silakan pilih gambar.", "danger")
        return redirect(url_for("index"))

    # Validasi: cek ekstensi file
    if not allowed_file(file.filename):
        flash(
            "Format file tidak didukung. Gunakan JPG, JPEG, atau PNG.",
            "danger"
        )
        return redirect(url_for("index"))

    # Simpan file dengan nama unik (mencegah konflik nama)
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(filepath)

    # Jalankan prediksi
    result = predict(filepath)

    if result["success"]:
        return render_template(
            "result.html",
            filename=unique_filename,
            class_name=result["class_name"],
            confidence=result["confidence"],
            all_predictions=result["all_predictions"],
        )
    else:
        flash(f"Gagal melakukan prediksi: {result['error']}", "danger")
        return redirect(url_for("index"))


@app.route("/about")
def about():
    """
    Halaman tentang aplikasi.
    """
    return render_template("index.html")


# ============================================
# Error Handlers
# ============================================

@app.errorhandler(413)
def file_too_large(error):
    """Handler untuk file yang terlalu besar (> 10 MB)."""
    flash("Ukuran file terlalu besar. Maksimal 10 MB.", "danger")
    return redirect(url_for("index"))


@app.errorhandler(404)
def page_not_found(error):
    """Handler untuk halaman tidak ditemukan."""
    flash("Halaman tidak ditemukan.", "warning")
    return redirect(url_for("index"))


# ============================================
# Entry Point
# ============================================
if __name__ == "__main__":
    print("=" * 55)
    print(" Butterfly Classification Web App")
    print(" Transfer Learning Xception + Flask")
    print("=" * 55)
    print(f" URL: http://127.0.0.1:5000")
    print("=" * 55)

    app.run(debug=True, host="0.0.0.0", port=5000)
