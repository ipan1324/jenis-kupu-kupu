"""
============================================
train.py - Script Training Model
============================================
Penerapan Transfer Learning Xception untuk
Klasifikasi Jenis Kupu-kupu

Mata Kuliah: AI - Bab 12 Transfer Learning
============================================
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dropout,
    Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================
# Konfigurasi
# ============================================
IMG_SIZE = (299, 299)          # Input size Xception
BATCH_SIZE = 16                # Batch size training
EPOCHS = 10                    # Jumlah epoch
NUM_CLASSES = 5                # Jumlah kelas kupu-kupu
LEARNING_RATE = 0.001          # Learning rate Adam

# Path dataset dan model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VALIDATION_DIR = os.path.join(BASE_DIR, "dataset", "validation")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "butterfly_xception.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")


def create_data_generators():
    """
    Membuat data generator untuk training, validasi, dan testing.

    Menggunakan ImageDataGenerator dengan augmentasi data
    untuk meningkatkan generalisasi model.

    Returns:
        tuple: (train_generator, validation_generator, test_generator)
    """
    # Data augmentation untuk training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,            # Normalisasi pixel ke [0, 1]
        rotation_range=20,            # Rotasi acak ±20 derajat
        zoom_range=0.2,               # Zoom acak ±20%
        horizontal_flip=True,         # Flip horizontal acak
        width_shift_range=0.1,        # Geser horizontal ±10%
        height_shift_range=0.1,       # Geser vertikal ±10%
        fill_mode="nearest"           # Isi pixel kosong
    )

    # Validasi dan test hanya perlu rescale
    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    print("\n[INFO] Memuat dataset training...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    print("[INFO] Memuat dataset validasi...")
    validation_generator = val_test_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    print("[INFO] Memuat dataset testing...")
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, validation_generator, test_generator


def build_model():
    """
    Membangun model Transfer Learning Xception.

    Arsitektur:
    - Base model: Xception (pretrained ImageNet)
    - GlobalAveragePooling2D
    - Dropout(0.5)
    - Dense(256, relu)
    - Dropout(0.3)
    - Dense(5, softmax)

    Returns:
        Model: Model Keras yang siap di-compile
    """
    print("\n[INFO] Membangun model Xception...")

    # Load Xception pretrained tanpa top layer
    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(299, 299, 3)
    )

    # Freeze seluruh base model (tidak ikut training)
    base_model.trainable = False
    print(f"[INFO] Base model Xception dimuat ({base_model.count_params():,} parameter)")
    print("[INFO] Seluruh base model di-freeze")

    # Tambahkan custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dropout(0.5, name="dropout_1")(x)
    x = Dense(256, activation="relu", name="dense_256")(x)
    x = Dropout(0.3, name="dropout_2")(x)
    predictions = Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    # Gabungkan base model dengan classification head
    model = Model(
        inputs=base_model.input,
        outputs=predictions,
        name="butterfly_xception"
    )

    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Tampilkan ringkasan arsitektur
    print("\n[INFO] Ringkasan Model:")
    print(f"  - Total parameter: {model.count_params():,}")
    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )
    print(f"  - Trainable parameter: {trainable:,}")
    print(f"  - Non-trainable parameter: {non_trainable:,}")

    return model


def save_labels(class_indices):
    """
    Simpan mapping label kelas ke file labels.txt.

    Args:
        class_indices (dict): Dictionary {nama_kelas: index}
    """
    # Balik mapping agar index -> nama_kelas
    labels = {v: k for k, v in class_indices.items()}

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for i in range(len(labels)):
            f.write(f"{labels[i]}\n")

    print(f"[INFO] Label kelas disimpan ke: {LABELS_PATH}")
    for idx, name in labels.items():
        print(f"  [{idx}] {name}")


def train():
    """
    Fungsi utama untuk menjalankan proses training.

    Pipeline:
    1. Buat data generator
    2. Bangun model Xception
    3. Setup callbacks (EarlyStopping, ModelCheckpoint)
    4. Jalankan training
    5. Evaluasi pada test set
    6. Simpan model dan label
    """
    print("=" * 55)
    print(" TRAINING: Transfer Learning Xception")
    print(" Klasifikasi Jenis Kupu-kupu")
    print("=" * 55)

    # Validasi keberadaan dataset
    for dir_name, dir_path in [
        ("Training", TRAIN_DIR),
        ("Validation", VALIDATION_DIR),
        ("Test", TEST_DIR),
    ]:
        if not os.path.exists(dir_path):
            print(f"\n[ERROR] Folder {dir_name} tidak ditemukan: {dir_path}")
            print("[INFO] Pastikan dataset sudah disiapkan di folder 'dataset/'")
            print("[INFO] Lihat README.md untuk panduan persiapan dataset.")
            return

    # Step 1: Buat data generator
    train_gen, val_gen, test_gen = create_data_generators()

    # Simpan label kelas
    save_labels(train_gen.class_indices)

    # Step 2: Bangun model
    model = build_model()

    # Step 3: Setup callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)

    callbacks = [
        # Hentikan training jika val_loss tidak membaik
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            verbose=1,
            restore_best_weights=True
        ),
        # Simpan model terbaik berdasarkan val_accuracy
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]

    # Step 4: Training
    print("\n" + "=" * 55)
    print(f" Memulai Training ({EPOCHS} epoch, batch_size={BATCH_SIZE})")
    print("=" * 55)

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Step 5: Evaluasi pada test set
    print("\n" + "=" * 55)
    print(" Evaluasi pada Test Set")
    print("=" * 55)

    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    print(f"\n  Test Loss    : {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Step 6: Simpan model final
    model.save(MODEL_PATH)
    print(f"\n[INFO] Model disimpan ke: {MODEL_PATH}")

    print("\n" + "=" * 55)
    print(" Training Selesai!")
    print("=" * 55)

    return history


# ============================================
# Entry Point
# ============================================
if __name__ == "__main__":
    train()
