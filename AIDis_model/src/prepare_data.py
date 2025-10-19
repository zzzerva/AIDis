"""
AIDis Makale Parse - Veri Hazırlama

Bu modül, ham veriyi model eğitimi için hazırlar.
"""
import random
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==== Ayarlar ====
ROOT = Path(__file__).resolve().parents[1]
RAW_NIST = ROOT / "data" / "raw" / "by_class"
RAW_DYS  = ROOT / "data" / "raw" / "dyslexic"
PROC     = ROOT / "data" / "processed"
TARGET_SIZE = (28, 28)
RANDOM_SEED = 42

# Denge ayarı: normal : dys oranını hedefle (ör: 1.0 => yaklaşık eşit)
BALANCE_NORMAL_TO_DYS = True
TARGET_NORMAL_TO_DYS_RATIO = 1.0
# ==================

def ensure_dirs():
    for split in ["train", "val", "test"]:
        for cls in ["normal", "dyslexic"]:
            (PROC / split / cls).mkdir(parents=True, exist_ok=True)

def gather_nist_images():
    imgs = []
    if not RAW_NIST.exists():
        print(f"UYARI: {RAW_NIST} bulunamadı.")
        return imgs
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for class_dir in RAW_NIST.iterdir():
        if class_dir.is_dir():
            files = [p for p in class_dir.rglob("*") if p.suffix.lower() in exts]
            imgs.extend([(f, "normal") for f in sorted(files)])
    return imgs

def gather_dys_images():
    if not RAW_DYS.exists():
        print(f"UYARI: {RAW_DYS} bulunamadı.")
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in RAW_DYS.rglob("*") if p.suffix.lower() in exts]
    return [(f, "dyslexic") for f in sorted(files)]

def pil_equalized_gray_resize(src_path):
    im  = Image.open(src_path).convert("L").resize(TARGET_SIZE)
    arr = np.array(im, dtype=np.uint8)
    arr = cv2.equalizeHist(arr)
    return Image.fromarray(arr)

def preprocess_and_split():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    ensure_dirs()

    normal_all = gather_nist_images()
    dys_all    = gather_dys_images()

    if len(normal_all) == 0 or len(dys_all) == 0:
        print("UYARI: normal veya dyslexic ham veri boş görünüyor.")
    print(f"Toplam ham: normal={len(normal_all)}  dyslexic={len(dys_all)}")

    # --- Sınıf dengesi (opsiyonel) ---
    if BALANCE_NORMAL_TO_DYS and len(dys_all) > 0:
        target_normal = int(len(dys_all) * TARGET_NORMAL_TO_DYS_RATIO)
        if target_normal < len(normal_all):
            random.shuffle(normal_all)
            normal_all = normal_all[:target_normal]
            print(f"Normal kırpıldı -> {len(normal_all)} (hedef oran {TARGET_NORMAL_TO_DYS_RATIO}:1)")

    all_items = normal_all + dys_all
    random.shuffle(all_items)

    n = len(all_items)
    if n == 0:
        print("Veri bulunamadı. Lütfen raw klasörlerini kontrol edin.")
        return

    n_train = int(0.7 * n)
    n_val   = int(0.1 * n)

    splits = {
        "train": all_items[:n_train],
        "val"  : all_items[n_train:n_train+n_val],
        "test" : all_items[n_train+n_val:]
    }

    # Makaledeki augmentasyon (yalnızca train)
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.10,
        zoom_range=0.05,
        fill_mode='nearest'
    )

    for split, items in splits.items():
        print(f"[{split}] yazılıyor... ({len(items)} örnek)")
        for src_path, label in items:
            try:
                im = pil_equalized_gray_resize(src_path)
                dst = PROC / split / label / f"{src_path.stem}_{random.randint(0,999999)}{src_path.suffix}"
                im.save(dst)

                if split == "train":
                    x = np.expand_dims(np.array(im, dtype=np.uint8), axis=(0, -1))  # (1,28,28,1)
                    aug = next(datagen.flow(x, batch_size=1))[0].astype(np.uint8).squeeze()
                    Image.fromarray(aug).save(dst.with_name(dst.stem + "_aug" + dst.suffix))
            except Exception as e:
                print("Hata:", src_path, e)

if __name__ == "__main__":
    preprocess_and_split()
    print("Veri hazırlandı:", PROC)
