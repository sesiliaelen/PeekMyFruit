"""
train.py
========
Melatih model klasifikasi kematangan buah dengan MobileNetV2 (transfer
learning) dalam dua fase:

    Fase 1 (feature extraction) : base MobileNetV2 dibekukan, hanya head
                                  (classifier) yang dilatih dengan LR besar.
    Fase 2 (fine-tuning)        : sebagian layer atas base di-unfreeze dan
                                  dilatih ulang dengan LR sangat kecil.

Setelah training:
    - Model disimpan ke models/peekmyfruit_model.keras
    - Label encoder disimpan ke models/label_encoder.json (+ .pkl)
    - Grafik akurasi/loss disimpan ke outputs/training_history.png

Jalankan:
    python src/train.py
"""

import json
import pickle

import matplotlib
matplotlib.use("Agg")  # backend non-interaktif supaya bisa save tanpa display
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import config

AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------------------------------------------------
# 1. DATASET
# ----------------------------------------------------------------------
def load_datasets():
    """Memuat train/val/test dari folder hasil preprocess.py.

    image_dataset_from_directory mengembalikan piksel float32 dalam
    rentang [0, 255]. Normalisasi ke rentang MobileNetV2 ([-1, 1])
    dilakukan DI DALAM model (lihat build_model) lewat preprocess_input,
    supaya preprocessing tidak pernah lupa diterapkan saat inferensi.
    """
    common = dict(
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int",        # label berupa integer index kelas
        seed=config.SEED,
    )
    train_ds = keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR, shuffle=True, **common
    )
    val_ds = keras.utils.image_dataset_from_directory(
        config.VAL_DIR, shuffle=False, **common
    )
    test_ds = keras.utils.image_dataset_from_directory(
        config.TEST_DIR, shuffle=False, **common
    )

    class_names = train_ds.class_names
    print(f"Kelas terdeteksi (urut index): {class_names}")

    # cache + prefetch untuk mempercepat I/O
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names


# ----------------------------------------------------------------------
# 2. DATA AUGMENTATION
# ----------------------------------------------------------------------
def build_augmentation():
    """Augmentasi sebagai layer Keras. Layer ini hanya aktif saat
    training (otomatis non-aktif saat inference), jadi aman dibungkus
    ke dalam model."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
            layers.RandomBrightness(0.1, value_range=(0, 255)),
        ],
        name="data_augmentation",
    )


# ----------------------------------------------------------------------
# 3. MODEL
# ----------------------------------------------------------------------
def build_model():
    """Bangun model fungsional: input -> augment -> preprocess ->
    MobileNetV2(base) -> GAP -> Dropout -> Dense(softmax)."""
    data_augmentation = build_augmentation()

    base_model = MobileNetV2(
        input_shape=config.IMG_SIZE + (3,),
        include_top=False,          # buang classifier ImageNet
        weights="imagenet",
    )
    base_model.trainable = False    # Fase 1: bekukan seluruh base

    inputs = keras.Input(shape=config.IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)         # skala [0,255] -> [-1,1] khas MobileNetV2
    # training=False menjaga BatchNorm tetap di mode inference (best practice)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(config.DROPOUT)(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="peekmyfruit")
    return model, base_model


# ----------------------------------------------------------------------
# 4. CALLBACKS
# ----------------------------------------------------------------------
def make_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]


# ----------------------------------------------------------------------
# 5. SIMPAN LABEL ENCODER
# ----------------------------------------------------------------------
def save_label_encoder(class_names):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # (a) JSON: mapping index <-> nama kelas (portable, mudah dibaca)
    mapping = {str(i): name for i, name in enumerate(class_names)}
    with open(config.LABELS_JSON, "w") as f:
        json.dump(mapping, f, indent=2)

    # (b) sklearn LabelEncoder (opsional, bila ingin API .transform/.inverse_transform)
    le = LabelEncoder()
    le.fit(class_names)
    with open(config.LABELS_PKL, "wb") as f:
        pickle.dump(le, f)

    print(f"Label encoder disimpan: {config.LABELS_JSON.name}, {config.LABELS_PKL.name}")


# ----------------------------------------------------------------------
# 6. PLOT HISTORY
# ----------------------------------------------------------------------
def plot_history(h1, h2):
    """Gabungkan history fase 1 & 2 lalu plot akurasi dan loss."""
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]
    split = len(h1.history["accuracy"])  # garis batas fase 1 -> 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(acc, label="train")
    ax1.plot(val_acc, label="val")
    ax1.axvline(split - 1, color="gray", ls="--", label="mulai fine-tune")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("epoch")
    ax1.legend()

    ax2.plot(loss, label="train")
    ax2.plot(val_loss, label="val")
    ax2.axvline(split - 1, color="gray", ls="--", label="mulai fine-tune")
    ax2.set_title("Loss")
    ax2.set_xlabel("epoch")
    ax2.legend()

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUTS_DIR / "training_history.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Grafik training disimpan: {out}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    keras.utils.set_random_seed(config.SEED)

    train_ds, val_ds, test_ds, class_names = load_datasets()
    save_label_encoder(class_names)

    model, base_model = build_model()

    # ---------- FASE 1: latih head ----------
    print("\n=== FASE 1: feature extraction (base dibekukan) ===")
    model.compile(
        optimizer=keras.optimizers.Adam(config.LR_HEAD),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_HEAD,
        callbacks=make_callbacks(),
    )

    # ---------- FASE 2: fine-tuning ----------
    print("\n=== FASE 2: fine-tuning sebagian base ===")
    base_model.trainable = True
    # Bekukan kembali semua layer kecuali FINE_TUNE_LAYERS layer terakhir
    for layer in base_model.layers[:-config.FINE_TUNE_LAYERS]:
        layer.trainable = False
    # BatchNorm dibiarkan beku agar statistik ImageNet tidak rusak
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(config.LR_FINETUNE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_FINETUNE,
        callbacks=make_callbacks(),
    )

    # ---------- Simpan model final ----------
    model.save(config.MODEL_PATH)
    print(f"\nModel final disimpan: {config.MODEL_PATH}")

    # ---------- Evaluasi cepat di test set ----------
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
    print("Jalankan `python src/evaluate.py` untuk metrik lengkap "
          "(precision/recall/F1/confusion matrix).")

    plot_history(history_head, history_ft)


if __name__ == "__main__":
    main()
