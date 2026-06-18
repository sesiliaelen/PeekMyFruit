"""
patch_model.py
==============
Memperbaiki file model .keras agar bisa dibuka oleh Keras versi lebih lama.

Model dilatih di Google Colab (Keras versi baru) yang menambahkan catatan
'quantization_config' pada layer. Catatan ini bernilai kosong (None) dan tidak
mempengaruhi cara kerja model sama sekali, tapi Keras versi lama tidak
mengenalinya sehingga gagal membuka model.

Script ini menghapus catatan tersebut dari file model. Kemampuan dan bobot
(weights) model tidak berubah sedikit pun.

Jalankan SEKALI saja:
    py patch_model.py
Lalu jalankan web app seperti biasa:
    py app.py
"""

import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "models" / "peekmyfruit_model.keras"
BACKUP = ROOT / "models" / "peekmyfruit_model_original.keras"


def strip_quant(obj):
    """Buang semua kunci 'quantization_config' di mana pun dalam config."""
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for v in obj.values():
            strip_quant(v)
    elif isinstance(obj, list):
        for v in obj:
            strip_quant(v)


def main():
    if not SRC.exists():
        raise SystemExit(f"Tidak menemukan file model di {SRC}")

    # 1. Backup file asli (sekali saja), untuk jaga-jaga
    if not BACKUP.exists():
        shutil.copy2(SRC, BACKUP)
        print(f"Backup file asli dibuat: {BACKUP.name}")

    # 2. File .keras sebenarnya adalah arsip zip. Baca semua isinya.
    with zipfile.ZipFile(SRC, "r") as z:
        items = {name: z.read(name) for name in z.namelist()}

    if "config.json" not in items:
        raise SystemExit("Struktur file .keras tidak seperti yang diharapkan "
                         "(config.json tidak ditemukan).")

    # 3. Bersihkan config.json
    cfg = json.loads(items["config.json"].decode("utf-8"))
    strip_quant(cfg)
    items["config.json"] = json.dumps(cfg).encode("utf-8")

    # 4. Tulis ulang file .keras dengan config yang sudah bersih
    with zipfile.ZipFile(SRC, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in items.items():
            z.writestr(name, data)
    print("Model berhasil dibersihkan.")

    # 5. Uji buka untuk memastikan sudah beres
    print("Menguji membuka model (sebentar)...")
    from tensorflow import keras
    keras.models.load_model(SRC)
    print("\nSUKSES! Model sekarang bisa dibuka.")
    print("Langkah berikutnya jalankan:  py app.py")


if __name__ == "__main__":
    main()
