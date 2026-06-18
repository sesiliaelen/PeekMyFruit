"""
config.py
=========
Konfigurasi terpusat untuk seluruh pipeline PeekMyFruit.
Diletakkan di satu tempat supaya train.py, evaluate.py, dan predict.py
selalu konsisten (ukuran gambar, path, nama kelas, dll).
"""

from pathlib import Path

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------
# ROOT = folder PeekMyFruit (satu level di atas folder src/)
ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"             # dataset asli dari Kaggle
PROCESSED_DIR = ROOT / "data" / "processed" # hasil split train/val/test
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"

MODEL_PATH = MODELS_DIR / "peekmyfruit_model.keras"
LABELS_JSON = MODELS_DIR / "label_encoder.json"      # mapping index -> nama kelas
LABELS_PKL = MODELS_DIR / "label_encoder.pkl"        # sklearn LabelEncoder (opsional)

# ----------------------------------------------------------------------
# KELAS (output kematangan) - urutan ini dipertahankan secara alfabet
# karena image_dataset_from_directory mengurutkan folder secara alfabet.
# ----------------------------------------------------------------------
CLASS_NAMES = ["ripe", "rotten", "unripe"]   # alfabetis -> index 0,1,2
NUM_CLASSES = len(CLASS_NAMES)

# ----------------------------------------------------------------------
# HYPERPARAMETER
# ----------------------------------------------------------------------
IMG_SIZE = (224, 224)        # input default MobileNetV2
BATCH_SIZE = 32
SEED = 42

# Rasio split (dari total data per kelas)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training
EPOCHS_HEAD = 15             # fase 1: latih head, base dibekukan
EPOCHS_FINETUNE = 15         # fase 2: fine-tuning sebagian base
LR_HEAD = 1e-3
LR_FINETUNE = 1e-5
FINE_TUNE_LAYERS = 40        # jumlah layer terakhir MobileNetV2 yang di-unfreeze
DROPOUT = 0.3
