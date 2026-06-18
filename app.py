"""
app.py
======
Web app PeekMyFruit. Menjalankan model .keras dan menyediakan halaman untuk
upload gambar atau ambil foto dari kamera, lalu menampilkan tingkat kematangan.

Cara menjalankan:
    pip install -r requirements.txt
    py app.py
Lalu buka browser ke:  http://127.0.0.1:5000
"""

import io
import json
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "peekmyfruit_model.keras"
LABELS_PATH = ROOT / "models" / "label_encoder.json"
IMG_SIZE = (224, 224)

# Nama tampilan yang ramah untuk tiap kelas
DISPLAY = {
    "unripe": "Mentah",
    "ripe": "Matang",
    "rotten": "Busuk",
}

app = Flask(__name__)

# ---- Muat model & label sekali saat server start ----
if not MODEL_PATH.exists():
    raise SystemExit(
        f"Model tidak ditemukan di {MODEL_PATH}.\n"
        "Download peekmyfruit_model.keras dari Google Drive (folder PeekMyFruit_hasil) "
        "lalu taruh di folder models/."
    )

print("Memuat model... (sebentar)")
model = keras.models.load_model(MODEL_PATH)
mapping = json.load(open(LABELS_PATH))
CLASS_NAMES = [mapping[str(i)] for i in range(len(mapping))]
print("Model siap. Kelas:", CLASS_NAMES)


def prepare(img: Image.Image):
    """Ubah gambar PIL jadi array [1,224,224,3] dengan piksel [0,255].
    Preprocessing MobileNetV2 sudah ada di dalam model, jadi cukup raw."""
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype="float32")
    return np.expand_dims(arr, axis=0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada gambar yang dikirim."}), 400
    try:
        raw = request.files["image"].read()
        img = Image.open(io.BytesIO(raw))
    except Exception:
        return jsonify({"error": "File gambar tidak bisa dibaca."}), 400

    probs = model.predict(prepare(img), verbose=0)[0]
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]

    return jsonify({
        "label": label,
        "display": DISPLAY.get(label, label.capitalize()),
        "confidence": float(probs[idx]),
        "all": [
            {"label": CLASS_NAMES[i],
             "display": DISPLAY.get(CLASS_NAMES[i], CLASS_NAMES[i].capitalize()),
             "prob": float(probs[i])}
            for i in range(len(CLASS_NAMES))
        ],
    })


if __name__ == "__main__":
    print("\nBuka browser ke:  http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
