"""
predict.py
==========
Inferensi tingkat kematangan untuk SATU gambar.

Model sudah memuat layer preprocessing (preprocess_input) di dalamnya,
jadi di sini cukup memuat gambar mentah dalam rentang [0, 255] lalu
mengirimkannya ke model.

Contoh pemakaian:
    python src/predict.py --image path/ke/gambar.jpg
    python src/predict.py --image apel.jpg --topk 3
"""

import argparse
import json

import numpy as np
from tensorflow import keras

import config


def load_labels():
    with open(config.LABELS_JSON) as f:
        mapping = json.load(f)
    return [mapping[str(i)] for i in range(len(mapping))]


def load_image(path):
    """Muat & resize gambar ke ukuran input model. Hasil: array [1,H,W,3]
    dengan piksel float32 dalam [0,255] (preprocessing di-handle model)."""
    img = keras.utils.load_img(path, target_size=config.IMG_SIZE)
    arr = keras.utils.img_to_array(img)          # [H,W,3], float [0,255]
    return np.expand_dims(arr, axis=0)           # tambah dimensi batch


def predict(image_path, topk=3):
    class_names = load_labels()
    model = keras.models.load_model(config.MODEL_PATH)

    x = load_image(image_path)
    probs = model.predict(x, verbose=0)[0]       # vektor probabilitas

    top_idx = np.argsort(probs)[::-1][:topk]

    print(f"\nGambar : {image_path}")
    print(f"Prediksi: {class_names[top_idx[0]].upper()} "
          f"({probs[top_idx[0]] * 100:.2f}%)\n")
    print("Detail probabilitas:")
    for i in top_idx:
        bar = "#" * int(probs[i] * 30)
        print(f"  {class_names[i]:8s} {probs[i] * 100:6.2f}%  {bar}")

    return class_names[top_idx[0]], float(probs[top_idx[0]])


def main():
    parser = argparse.ArgumentParser(description="PeekMyFruit - prediksi 1 gambar")
    parser.add_argument("--image", required=True, help="path gambar buah")
    parser.add_argument("--topk", type=int, default=3, help="tampilkan top-K kelas")
    args = parser.parse_args()

    if not config.MODEL_PATH.exists():
        raise SystemExit(f"Model tidak ditemukan di {config.MODEL_PATH}. "
                         "Jalankan train.py dulu.")

    predict(args.image, args.topk)


if __name__ == "__main__":
    main()
