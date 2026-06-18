"""
preprocess.py
=============
Mengubah dataset mentah Kaggle ("Fruit Ripeness: Unripe, Ripe and Rotten")
menjadi struktur folder yang siap dilatih:

    data/processed/
        train/  {ripe, rotten, unripe}/
        val/    {ripe, rotten, unripe}/
        test/   {ripe, rotten, unripe}/

Catatan penting
---------------
Dataset Kaggle bisa punya struktur folder bermacam-macam, misalnya:
    raw/Apple/ripe/...        raw/unripe_banana/...        raw/Orange Rotten/...
Script ini TIDAK peduli nama buah. Ia hanya membaca KATA KUNCI kematangan
("unripe", "ripe", "rotten") dari seluruh path setiap gambar, lalu
menggabungkan ketiga buah (apple, banana, orange) ke dalam 3 kelas saja.

Karena "ripe" adalah substring dari "unripe", urutan pengecekan penting:
cek "unripe" dulu, baru "rotten", baru "ripe".

Jalankan:
    python src/preprocess.py
"""

import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

import config

# Ekstensi gambar yang diterima
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def infer_class(path_str: str) -> str | None:
    """Tentukan kelas kematangan dari string path (case-insensitive).

    Urutan pengecekan penting: 'unripe' diperiksa sebelum 'ripe'
    karena 'ripe' adalah substring dari 'unripe'.
    """
    p = path_str.lower()
    if "unripe" in p:
        return "unripe"
    if "rotten" in p:
        return "rotten"
    if "ripe" in p:
        return "ripe"
    return None  # gambar yang tak bisa diklasifikasi -> dilewati


def collect_images(raw_dir: Path):
    """Telusuri semua gambar di raw_dir dan pasangkan dengan kelasnya.

    Returns: dict {class_name: [Path, Path, ...]}
    """
    buckets = {c: [] for c in config.CLASS_NAMES}
    skipped = 0

    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMG_EXT:
            continue
        label = infer_class(str(path))
        if label is None:
            skipped += 1
            continue
        buckets[label].append(path)

    print(f"  Gambar dilewati (tak terklasifikasi): {skipped}")
    return buckets


def split_files(files, seed):
    """Stratified split satu daftar file menjadi (train, val, test)."""
    # Pertama pisahkan train vs sisanya (val + test)
    train, temp = train_test_split(
        files, train_size=config.TRAIN_RATIO, random_state=seed, shuffle=True
    )
    # Bagi sisanya menjadi val dan test secara proporsional
    rel_val = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val, test = train_test_split(
        temp, train_size=rel_val, random_state=seed, shuffle=True
    )
    return train, val, test


def copy_files(files, dest_dir: Path):
    """Salin daftar file ke folder tujuan (nama file diberi prefix unik
    agar tak bentrok antar buah)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files):
        # prefix angka mencegah tabrakan nama (mis. banyak file bernama '1.jpg')
        dest = dest_dir / f"{i:05d}_{src.name}"
        shutil.copy2(src, dest)


def main():
    print(f"[1/3] Membaca dataset mentah dari: {config.RAW_DIR}")
    if not config.RAW_DIR.exists() or not any(config.RAW_DIR.iterdir()):
        raise SystemExit(
            f"Folder {config.RAW_DIR} kosong.\n"
            "Letakkan/unzip dataset Kaggle ke dalam data/raw/ terlebih dahulu."
        )

    buckets = collect_images(config.RAW_DIR)
    for c in config.CLASS_NAMES:
        print(f"    {c:8s}: {len(buckets[c])} gambar")

    # Bersihkan processed lama supaya tidak ada sisa
    if config.PROCESSED_DIR.exists():
        shutil.rmtree(config.PROCESSED_DIR)

    print("[2/3] Split stratified (train/val/test) per kelas...")
    summary = {}
    for c in config.CLASS_NAMES:
        if len(buckets[c]) == 0:
            print(f"    PERINGATAN: kelas '{c}' tidak punya gambar, dilewati.")
            continue
        train, val, test = split_files(buckets[c], config.SEED)
        copy_files(train, config.TRAIN_DIR / c)
        copy_files(val, config.VAL_DIR / c)
        copy_files(test, config.TEST_DIR / c)
        summary[c] = (len(train), len(val), len(test))

    print("[3/3] Selesai. Ringkasan jumlah file:")
    print(f"    {'kelas':8s}  {'train':>6s} {'val':>5s} {'test':>5s}")
    for c, (tr, va, te) in summary.items():
        print(f"    {c:8s}  {tr:6d} {va:5d} {te:5d}")
    print(f"\n    Output ada di: {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
