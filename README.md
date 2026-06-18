# 🍎🍌🍊 PeekMyFruit

Model AI untuk mendeteksi tingkat kematangan buah (**Unripe / Ripe / Rotten**)
dari gambar, menggunakan **MobileNetV2 transfer learning** (TensorFlow/Keras).

Buah yang didukung: Apple, Banana, Orange — semuanya digabung ke dalam
3 kelas kematangan (model fokus pada kematangan, bukan jenis buah).

---

## 📁 Struktur Folder

```
PeekMyFruit/
├── data/
│   ├── raw/                 # <-- letakkan dataset Kaggle mentah di sini
│   └── processed/           # otomatis dibuat oleh preprocess.py
│       ├── train/  {ripe, rotten, unripe}/
│       ├── val/    {ripe, rotten, unripe}/
│       └── test/   {ripe, rotten, unripe}/
├── models/
│   ├── peekmyfruit_model.keras   # model terlatih
│   ├── label_encoder.json        # mapping index -> nama kelas
│   └── label_encoder.pkl         # sklearn LabelEncoder (opsional)
├── outputs/
│   ├── training_history.png      # grafik akurasi/loss
│   └── confusion_matrix.png      # confusion matrix
├── src/
│   ├── config.py            # konfigurasi terpusat (path, hyperparameter)
│   ├── preprocess.py        # konsolidasi + split dataset
│   ├── train.py             # training 2 fase + simpan model
│   ├── evaluate.py          # metrik lengkap di test set
│   └── predict.py           # inferensi 1 gambar
├── notebooks/               # (opsional) eksperimen
├── requirements.txt
└── README.md
```

---

## 🚀 Cara Pakai

### 1. Setup environment
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Siapkan dataset
Download dataset Kaggle *"Fruit Ripeness: Unripe, Ripe and Rotten"*,
lalu unzip semua isinya ke dalam folder `data/raw/`.

Struktur folder asli tidak masalah — `preprocess.py` membaca kata kunci
`unripe` / `ripe` / `rotten` dari nama folder/file untuk menentukan kelas.

### 3. Preprocessing + split
```bash
python src/preprocess.py
```

### 4. Training
```bash
python src/train.py
```
Menghasilkan `models/peekmyfruit_model.keras`, label encoder, dan
`outputs/training_history.png`.

### 5. Evaluasi
```bash
python src/evaluate.py
```
Menampilkan Accuracy, Precision, Recall, F1, dan Confusion Matrix.

### 6. Prediksi 1 gambar
```bash
python src/predict.py --image contoh.jpg
```

---

## 🧠 Catatan Teknis (best practice)

- **Preprocessing di dalam model**: layer `preprocess_input` MobileNetV2
  ditanam ke dalam model, jadi `predict.py` cukup memberi gambar mentah
  `[0,255]` — tidak ada risiko normalisasi yang lupa diterapkan.
- **Augmentasi sebagai layer Keras** (`RandomFlip`, `RandomRotation`, dll.)
  hanya aktif saat training, otomatis non-aktif saat inferensi.
- **Dua fase training**: feature extraction (base beku) → fine-tuning
  (unfreeze sebagian layer atas, LR sangat kecil). BatchNorm tetap dibekukan
  saat fine-tuning agar statistik ImageNet tidak rusak.
- **Format `.keras`**: format penyimpanan native Keras 3 terbaru.
- **Metrik `macro`**: precision/recall/F1 dirata-rata per kelas dengan
  bobot sama, cocok bila jumlah data antar kelas tidak seimbang.
```
```
