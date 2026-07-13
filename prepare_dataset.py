"""
============================================
prepare_dataset.py - Persiapan Dataset
============================================
Script untuk menyiapkan dataset dari format
Kaggle ke struktur folder yang dibutuhkan
oleh ImageDataGenerator.

Hanya mengambil 5 kelas:
- Monarch
- Peacock
- Julia
- Viceroy
- Zebra Long Wing
============================================
"""

import os
import csv
import shutil
import random

# ============================================
# Konfigurasi
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path archive Kaggle
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
ARCHIVE_TRAIN_DIR = os.path.join(ARCHIVE_DIR, "train")
TRAINING_CSV = os.path.join(ARCHIVE_DIR, "Training_set.csv")

# Path dataset output
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALIDATION_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# 5 kelas yang digunakan (sesuaikan dengan label di CSV)
# Label di CSV menggunakan UPPERCASE
SELECTED_CLASSES = {
    "MONARCH": "Monarch",
    "PEACOCK": "Peacock",
    "JULIA": "Julia",
    "VICEROY": "Viceroy",
    "ZEBRA LONG WING": "Zebra Long Wing",
}

# Rasio split: 80% train, 10% validation, 10% test
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# Random seed untuk reprodusibilitas
RANDOM_SEED = 42


def read_csv_mapping(csv_path):
    """
    Membaca file CSV dan mengembalikan mapping filename -> label.

    Args:
        csv_path (str): Path ke file CSV

    Returns:
        dict: {filename: label}
    """
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].strip()
            label = row["label"].strip()
            mapping[filename] = label
    return mapping


def prepare_dataset():
    """
    Menyiapkan dataset dari format Kaggle ke struktur folder.

    Pipeline:
    1. Baca CSV mapping
    2. Filter hanya 5 kelas yang dipilih
    3. Buat folder per kelas di train/, validation/, test/
    4. Copy gambar training (split 80/10/10 ke train/validation/test)
    """
    print("=" * 55)
    print(" Persiapan Dataset Kupu-kupu")
    print(" Mengambil 5 kelas dari dataset Kaggle")
    print("=" * 55)

    # Validasi file CSV
    if not os.path.exists(TRAINING_CSV):
        print(f"\n[ERROR] File tidak ditemukan: {TRAINING_CSV}")
        print("[INFO] Pastikan dataset Kaggle sudah diekstrak di folder 'archive/'")
        return

    # Step 1: Baca CSV mapping
    print("\n[INFO] Membaca file CSV...")
    train_mapping = read_csv_mapping(TRAINING_CSV)

    print(f"  Total gambar di CSV: {len(train_mapping)}")

    # Step 2: Filter 5 kelas yang dipilih
    print(f"\n[INFO] Memfilter {len(SELECTED_CLASSES)} kelas yang dipilih...")

    train_filtered = {}
    for filename, label in train_mapping.items():
        if label in SELECTED_CLASSES:
            folder_name = SELECTED_CLASSES[label]
            if folder_name not in train_filtered:
                train_filtered[folder_name] = []
            train_filtered[folder_name].append(filename)

    print("\n  Distribusi data per kelas:")
    total_data = 0
    for class_name, files in sorted(train_filtered.items()):
        print(f"    {class_name:20s}: {len(files)} gambar")
        total_data += len(files)
    print(f"    {'TOTAL':20s}: {total_data} gambar")

    # Step 3: Buat folder struktur
    print("\n[INFO] Membuat struktur folder dataset...")

    for class_name in SELECTED_CLASSES.values():
        os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(VALIDATION_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, class_name), exist_ok=True)

    # Step 4: Copy gambar training (split train/validation/test)
    print(f"\n[INFO] Menyalin gambar (split 80/10/10)...")

    random.seed(RANDOM_SEED)
    train_count = 0
    val_count = 0
    test_count = 0

    for class_name, files in sorted(train_filtered.items()):
        # Shuffle dan split
        random.shuffle(files)
        train_idx = int(len(files) * TRAIN_SPLIT)
        val_idx = train_idx + int(len(files) * VAL_SPLIT)
        
        train_files = files[:train_idx]
        val_files = files[train_idx:val_idx]
        test_files = files[val_idx:]

        # Copy ke train/
        for f in train_files:
            src = os.path.join(ARCHIVE_TRAIN_DIR, f)
            dst = os.path.join(TRAIN_DIR, class_name, f)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                train_count += 1

        # Copy ke validation/
        for f in val_files:
            src = os.path.join(ARCHIVE_TRAIN_DIR, f)
            dst = os.path.join(VALIDATION_DIR, class_name, f)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                val_count += 1

        # Copy ke test/
        for f in test_files:
            src = os.path.join(ARCHIVE_TRAIN_DIR, f)
            dst = os.path.join(TEST_DIR, class_name, f)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                test_count += 1

        print(f"  {class_name:20s}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    # Summary
    print("\n" + "=" * 55)
    print(" Dataset Siap!")
    print("=" * 55)
    print(f"  Training   : {train_count} gambar")
    print(f"  Validation : {val_count} gambar")
    print(f"  Testing    : {test_count} gambar")
    print(f"  Total      : {train_count + val_count + test_count} gambar")
    print(f"\n  Lokasi: {DATASET_DIR}")
    print("\n[INFO] Selanjutnya jalankan: python train.py")


# ============================================
# Entry Point
# ============================================
if __name__ == "__main__":
    prepare_dataset()
