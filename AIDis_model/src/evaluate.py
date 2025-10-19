"""
AIDis Makale Parse - Model Değerlendirme

Bu modül, eğitilmiş modeli test verileri üzerinde değerlendirir ve
kelime görsellerini harflere bölerek sınıflandırır.
"""
from pathlib import Path
import json, csv, shutil
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score

# Modüler import'lar
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config, get_model_path, get_class_indices
from utils import imread_unicode, imwrite_unicode, equalize_resize_from_gray, segment_letters, sort_boxes_reading_order

# Konfigürasyonu yükle
config = get_config()


def load_class_indices():
    """Sınıf haritasını yükle"""
    return get_class_indices(), {v: k for k, v in get_class_indices().items()}

def list_test_files(class_indices):
    """Test dosyalarını listele"""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files, labels = [], []
    test_dir = config["TEST_DIR"]
    
    for cls_name, cls_idx in sorted(class_indices.items(), key=lambda x: x[1]):
        d = test_dir / cls_name
        if not d.exists():
            print(f"[UYARI] {d} yok, atlanıyor."); continue
        for p in d.rglob("*"):
            if p.suffix.lower() in exts:
                files.append(p); labels.append(cls_idx)
    return files, np.array(labels, dtype=np.int32)

# ========== Önişleme (EĞİTİMLE AYNI) ==========
def preprocess_strict(path: Path) -> np.ndarray:
    """Strict preprocessing (eğitimle birebir aynı)"""
    im = Image.open(path).convert("L").resize(config["IMG_SIZE"])
    arr = np.array(im, dtype=np.uint8)
    arr = cv2.equalizeHist(arr)
    return arr

def _invert(arr: np.ndarray) -> np.ndarray:
    """Görüntüyü ters çevir"""
    return 255 - arr

def _to_input(arr: np.ndarray) -> np.ndarray:
    """CNN için input formatına çevir"""
    return (arr.astype(np.float32) / 255.0)[..., None]
# ==============================================

def _tta_variants(path: Path):
    """TTA (Test Time Augmentation) varyantları oluştur"""
    variants = []
    if config["TTA_USE_STRICT"]:
        s = preprocess_strict(path)
        variants.append(s)
        if config["TTA_USE_INVERT"]: 
            variants.append(_invert(s))
    return variants

# ========== TEST IMAGES DEĞERLENDİR ==========
def evaluate_test_images(model, class_indices, inv_map):
    X_paths, y_true = list_test_files(class_indices)
    if len(X_paths) == 0:
        print("Test görseli bulunamadı. test_images/normal & dyslexic doldur.")
        return

    # TTA: çoklu önişleme -> model -> reduce
    dbg_dir = config["ROOT"] / "debug_eval_28x28"
    if config["SAVE_DEBUG_28x28"]:
        if dbg_dir.exists(): shutil.rmtree(dbg_dir)
        dbg_dir.mkdir(parents=True, exist_ok=True)

    y_prob = []
    for p in X_paths:
        arrs = _tta_variants(p)
        if config["SAVE_DEBUG_28x28"]:
            for i, a in enumerate(arrs):
                cv2.imwrite(str(dbg_dir / f"{p.relative_to(config['TEST_DIR']).as_posix().replace('/','_')}_v{i}.png"), a)
        Xv = np.stack([_to_input(a) for a in arrs], axis=0)
        pv = model.predict(Xv, verbose=0).ravel()
        y_prob.append(pv.mean() if config["TTA_REDUCER"]=="avg" else pv.max())
    y_prob = np.array(y_prob, dtype=np.float32)
    y_pred = (y_prob >= config["THRESHOLD"]).astype(np.int32)

    # Satır satır çıktı + CSV + yanlış kopyalama
    rows = []
    mis_dir = config["ROOT"] / "misclassified_test_images"
    if config["SAVE_MISCLASSIFIED"]:
        if mis_dir.exists(): shutil.rmtree(mis_dir)
        mis_dir.mkdir(parents=True, exist_ok=True)

    note = f"P({inv_map.get(1, 'class1')})"
    for i, p in enumerate(X_paths):
        t = int(y_true[i]); pr = int(y_pred[i]); prob = float(y_prob[i])
        tname, pname = inv_map[t], inv_map[pr]
        ok = (t==pr)
        if config["PRINT_PER_SAMPLE"]:
            print((p.relative_to(config["TEST_DIR"]).as_posix(), prob, pname, tname, "✓" if ok else "✗"))
        rows.append({"file": p.relative_to(config["TEST_DIR"]).as_posix(),
                     "true_label": tname, "pred_label": pname, note: prob, "correct": int(ok)})
        if config["SAVE_MISCLASSIFIED"] and not ok:
            shutil.copy2(p, mis_dir / p.name)

    if config["SAVE_CSV"]:
        out_csv = config["TEST_PRED_CSV"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
        print("CSV kaydedildi:", out_csv)
        if config["SAVE_MISCLASSIFIED"]: print("Yanlış sınıflananlar:", mis_dir)

    # Metrikler @ varsayılan eşik
    cm = confusion_matrix(y_true, y_pred, labels=[class_indices.get("dyslexic",0),
                                                  class_indices.get("normal",1)])
    print("\n=== Varsayılan Eşik = %.2f ===" % config["THRESHOLD"])
    print("Confusion Matrix (rows=true, cols=pred):\n", cm)
    tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
    acc  = (tp + tn) / max(1, cm.sum())
    sens = tp / max(1, (tp + fn))  # recall(normal)
    spec = tn / max(1, (tn + fp))  # recall(dyslexic)
    try: auc = roc_auc_score(y_true, y_prob)
    except: auc = float('nan')
    print(f"Accuracy: {acc:.4f} | Sens(1): {sens:.4f} | Spec(0): {spec:.4f} | AUC: {auc:.4f}")

    names = [k for k,_ in sorted(class_indices.items(), key=lambda x: x[1])]
    print("\nClassification Report:\n",
          classification_report(y_true, y_pred, target_names=names, zero_division=0))

    # Eşik taraması (macro-F1 en iyi)
    thr_grid = np.unique(np.concatenate([np.linspace(0,1,201), np.round(y_prob,6)]))
    best = {"thr":0.5,"macro_f1":-1,"cm":None,"acc":0,"sens":0,"spec":0}
    for thr in thr_grid:
        y_hat = (y_prob >= thr).astype(int)
        cm_t = confusion_matrix(y_true, y_hat, labels=[0,1])
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        f1_macro = f1_score(y_true, y_hat, average="macro", zero_division=0)
        if f1_macro > best["macro_f1"]:
            best = {"thr": float(thr), "macro_f1": float(f1_macro),
                    "cm": cm_t, "acc": float((tp_t+tn_t)/cm_t.sum()),
                    "sens": float(tp_t/max(1,(tp_t+fn_t))),
                    "spec": float(tn_t/max(1,(tn_t+fp_t)))}
    print("\n[THRESHOLD SEARCH] Strateji: macro_f1 -> En iyi eşik: %.3f" % best["thr"])
    print("Confusion Matrix @thr:\n", best["cm"])
    print("Accuracy: %.4f  Sensitivity: %.4f  Specificity: %.4f  Macro-F1: %.4f" %
          (best["acc"], best["sens"], best["spec"], best["macro_f1"]))

# ========== KELİME/CÜMLE -> HARF PARSER ==========
def _sort_boxes_reading_order(boxes):
    # boxes: list of (x,y,w,h)
    if not boxes: return []
    hs = [h for (_,_,_,h) in boxes]
    med_h = int(np.median(hs)) if hs else 20
    line_h = max(12, int(0.6 * med_h))

    def key(b):
        x,y,w,h = b
        row = y // line_h
        return (row, x)
    return sorted(boxes, key=key)

def _segment_letters(gray):
    """
    Girdi: gri kelime/cümle görüntüsü.
    Çıktı: reading-order sıralı bounding box listesi [(x,y,w,h), ...]
    """
    H, W = gray.shape
    eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(eq, (3,3), 0)
    # Harfler beyaz olacak şekilde binary INV + Otsu
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # küçük gürültüleri temizle (açma)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    # harf boşluklarını çok kapatmadan hafif genişlet
    bw = cv2.dilate(bw, np.ones((2,2), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    min_area = max(20, int(0.0015 * H * W))     # çok küçük gürültüleri at
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area < min_area:       # çok küçük
            continue
        if h < 5 or w < 3:        # çok ince
            continue
        if h > 0.9*H and w < 0.1*W:
            continue              # dikey büyük parazit
        boxes.append((x,y,w,h))

    boxes = _sort_boxes_reading_order(boxes)
    return boxes

def run_letter_parser_and_classify(model, inv_map):
    """Kelimeden harfe kırpma ve sınıflandırma"""
    if not config["LETTER_PARSER_ENABLED"]:
        return
    if not config["WORD_DIR"].exists():
        print(f"[Letter Parser] {config['WORD_DIR']} yok; atlanıyor.")
        return

    if config["CROPS_DIR"].exists(): shutil.rmtree(config["CROPS_DIR"])
    config["CROPS_DIR"].mkdir(parents=True, exist_ok=True)
    if config["DRAW_ANNOTATED"]:
        if config["ANNOT_DIR"].exists(): shutil.rmtree(config["ANNOT_DIR"])
        config["ANNOT_DIR"].mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(config["WORD_DIR"].glob("*")):
        if p.suffix.lower() not in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}:
            continue

        img = imread_unicode(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: 
            print("[Parser] Okunamadı:", p.name); 
            continue

        boxes = segment_letters(img)
        if not boxes:
            print("[Parser] Harf bulunamadı:", p.name)
            continue

        ann = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if config["DRAW_ANNOTATED"] else None

        for i,(x,y,w,h) in enumerate(boxes):
            pad = max(1, int(0.08 * max(w,h)))
            x0, y0 = max(0, x-pad), max(0, y-pad)
            x1, y1 = min(img.shape[1], x+w+pad), min(img.shape[0], y+h+pad)
            crop_gray = img[y0:y1, x0:x1]

            # EĞİTİMLE AYNI ÖNİŞLEME
            letter28 = equalize_resize_from_gray(crop_gray)
            out_path = config["CROPS_DIR"] / f"{p.stem}_char{i:03d}.png"
            imwrite_unicode(str(out_path), letter28)

            X = (letter28.astype(np.float32)/255.0)[None, ..., None]
            prob = float(model.predict(X, verbose=0).ravel()[0])
            pred_idx = int(prob >= config["THRESHOLD"])
            pred_label = inv_map[pred_idx]

            rows.append({
                "word_file": p.name,
                "index": i,
                "bbox": f"{x},{y},{w},{h}",
                f"P({inv_map.get(1,'class1')})": prob,
                "pred_label": pred_label,
                "crop_file": out_path.name
            })

            if config["DRAW_ANNOTATED"]:
                color = (0, 200, 0) if pred_label == "normal" else (0, 0, 255)
                cv2.rectangle(ann, (x0,y0), (x1,y1), color, 2)
                cv2.putText(ann, pred_label, (x0, max(10,y0-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if config["DRAW_ANNOTATED"]:
            imwrite_unicode(str(config["ANNOT_DIR"] / f"{p.stem}_annot.png"), ann)

    if rows:
        with open(config["CROPS_PRED_CSV"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
        print("Letter crops klasörü:", config["CROPS_DIR"])
        print("Letter crop tahminleri CSV:", config["CROPS_PRED_CSV"])
        if config["DRAW_ANNOTATED"]:
            print("Anotasyonlu çıktılar:", config["ANNOT_DIR"])
# ============================================================

def main():
    """Ana fonksiyon"""
    model_path = get_model_path()
    if not model_path.exists():
        print("Model bulunamadı:", model_path); return
    model = tf.keras.models.load_model(str(model_path))

    class_indices, inv_map = load_class_indices()
    print("TRAIN class_indices:", class_indices)
    print(f"Prob yorumu: P({inv_map.get(1,'class1')}) (eğitimde normal=1)")

    # 1) Tek-harf test seti değerlendirmesi (mevcut akışın değişmedi)
    evaluate_test_images(model, class_indices, inv_map)

    # 2) Kelime/cümle görsellerini harflere böl + sınıflandır
    run_letter_parser_and_classify(model, inv_map)

if __name__ == "__main__":
    main()
