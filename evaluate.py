"""
evaluate.py
===========
Mengevaluasi model terlatih pada test set dan menampilkan:
    - Accuracy
    - Precision, Recall, F1-score (per kelas & rata-rata)
    - Confusion Matrix (dicetak + disimpan sebagai gambar)

Jalankan:
    python src/evaluate.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow import keras

import config


def load_labels():
    """Baca mapping index -> nama kelas dari label_encoder.json."""
    with open(config.LABELS_JSON) as f:
        mapping = json.load(f)
    # urutkan berdasarkan index agar konsisten dengan output model
    return [mapping[str(i)] for i in range(len(mapping))]


def main():
    if not config.MODEL_PATH.exists():
        raise SystemExit(f"Model tidak ditemukan di {config.MODEL_PATH}. "
                         "Jalankan train.py dulu.")

    class_names = load_labels()
    model = keras.models.load_model(config.MODEL_PATH)

    # Test set TIDAK di-shuffle agar urutan label & prediksi sinkron
    test_ds = keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    # Kumpulkan label asli dan prediksi
    y_true, y_pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ---------- Metrik ----------
    acc = accuracy_score(y_true, y_pred)
    # 'macro' = rata-rata tiap kelas dengan bobot sama (bagus utk data tak seimbang)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print("\n================ HASIL EVALUASI ================")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall    (macro): {rec:.4f}")
    print(f"F1-score  (macro): {f1:.4f}")
    print("\nLaporan per kelas:")
    print(classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    ))

    # ---------- Confusion Matrix ----------
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix (baris = aktual, kolom = prediksi):")
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix - PeekMyFruit")
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUTS_DIR / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\nConfusion matrix disimpan: {out}")


if __name__ == "__main__":
    main()
