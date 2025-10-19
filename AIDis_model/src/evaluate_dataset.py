"""
AIDis Makale Parse - Dataset Değerlendirme

Bu modül, test dataset'i üzerinde model performansını değerlendirir.
"""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score

ROOT = Path(r"C:/AIDis_makale")
PROC = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "best_model.keras"
CLASS_IDX_PATH = ROOT / "class_indices.json"
IMG_SIZE = (28, 28)
BATCH_SIZE = 64
THRESHOLD = 0.5

def main():
    if not MODEL_PATH.exists():
        print("Model bulunamadı:", MODEL_PATH); return
    if not (PROC / "test").exists():
        print("Test klasörü bulunamadı:", PROC / "test"); return

    if CLASS_IDX_PATH.exists():
        with open(CLASS_IDX_PATH, "r") as f:
            class_indices = json.load(f)
    else:
        class_indices = {"dyslexic":0, "normal":1}
        print("UYARI: class_indices.json yok; varsayılan:", class_indices)

    model = tf.keras.models.load_model(str(MODEL_PATH))
    test_gen = ImageDataGenerator(rescale=1./255.0).flow_from_directory(
        PROC / "test",
        target_size=IMG_SIZE,
        color_mode="grayscale",
        class_mode="binary",
        shuffle=False,
        batch_size=BATCH_SIZE
    )

    print("Test class_indices:", test_gen.class_indices)

    y_true = test_gen.classes.astype(int)
    y_prob = model.predict(test_gen, verbose=0).ravel()
    y_pred = (y_prob >= THRESHOLD).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    print("\nConfusion Matrix (rows=true, cols=pred):\n", cm)
    tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)

    acc  = (tp + tn) / max(1, cm.sum())
    sens = tp / max(1, (tp + fn))  # recall of class '1'
    spec = tn / max(1, (tn + fp))
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')

    # macro-F1
    try:
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    except Exception:
        macro_f1 = float('nan')

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Sensitivity (Recall for '1'): {sens:.4f}")
    print(f"Specificity (Recall for '0'): {spec:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    names = [k for k,_ in sorted(test_gen.class_indices.items(), key=lambda x:x[1])]
    print("\nClassification Report:\n",
          classification_report(y_true, y_pred, target_names=names, zero_division=0))

    # sınıf bazında özet
    inv = {v:k for k,v in test_gen.class_indices.items()}
    print("\nSınıf bazında özet:")
    for idx in sorted(inv.keys()):
        cls = inv[idx]
        mask = (y_true==idx)
        total = int(mask.sum())
        correct = int(((y_pred==idx) & mask).sum())
        rec = (correct/total) if total>0 else 0.0
        print(f"  {cls}: toplam {total}, doğru {correct}, yanlış {total-correct} (Recall={rec:.2%})")

if __name__ == "__main__":
    main()
