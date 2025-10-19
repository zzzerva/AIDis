# core/utils.py
import os
import cv2
import numpy as np
from django.conf import settings
import tensorflow as tf

# ================== Ayarlanabilir Parametreler (ENV ile override) ==================
AIDIS_IMG_SIZE       = int(os.getenv("AIDIS_IMG_SIZE", 28))
AIDIS_BLOCK          = int(os.getenv("AIDIS_BLOCK", 31))
AIDIS_C              = int(os.getenv("AIDIS_C", 5))

# Segmentasyon defaultları (FP azaltma odaklı)
AIDIS_MIN_AREA       = float(os.getenv("AIDIS_MIN_AREA_RATIO", 0.0014))  # min kutu alan oranı (H*W ile çarpılır)
AIDIS_MIN_GAP_FR     = float(os.getenv("AIDIS_MIN_GAP_FR", 0.10))        # dikey projede boşluk eşiği (0..1)
AIDIS_MIN_HEIGHT_FR  = float(os.getenv("AIDIS_MIN_HEIGHT_FR", 0.40))     # medyan kutu yüksekliğinin min oranı
AIDIS_MAX_W_RATIO    = float(os.getenv("AIDIS_MAX_W_RATIO", 0.20))       # satırı saran aşırı geniş kutu eşiği
AIDIS_SPLIT_W_MUL    = float(os.getenv("AIDIS_SPLIT_W_MUL", 2.0))        # medyan genişliğin katı ise böl
AIDIS_DESKEW_MAX     = float(os.getenv("AIDIS_DESKEW_MAX", 15.0))

# Tahmin – “dyslexic” için sıkı kural (binary çıktı, uncertain YOK)
AIDIS_USE_INVERT_TTA = os.getenv("AIDIS_TTA_INVERT", "1").strip() in {"1","true","True"}
AIDIS_DYS_MIN        = float(os.getenv("AIDIS_DYS_MIN", 0.80))           # p_dys >= 0.80
AIDIS_MARGIN_MIN     = float(os.getenv("AIDIS_MARGIN_MIN", 0.20))        # p_dys - p_norm >= 0.20
# ===================================================================================

# Model yol seçimi
CANDIDATES = [
    os.path.join(settings.BASE_DIR, "models", "final_model.keras"),
    os.path.join(settings.BASE_DIR, "models", "final_model.h5"),
    os.path.join(settings.BASE_DIR, "models", "final_model"),
]
_model = None

def _pick_model_path():
    for p in CANDIDATES:
        if os.path.exists(p):
            return p
    models_dir = os.path.join(settings.BASE_DIR, "models")
    listing = os.listdir(models_dir) if os.path.isdir(models_dir) else []
    raise FileNotFoundError(
        "Model dosyası bulunamadı.\nAranan yollar:\n - " + "\n - ".join(CANDIDATES) +
        f"\n'models' klasörü içeriği: {listing}"
    )

def get_model():
    global _model
    if _model is None:
        model_path = _pick_model_path()
        print(f"[AIDis] Loading model from: {model_path}")
        _model = tf.keras.models.load_model(model_path, compile=not os.path.isdir(model_path))
    return _model

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# Unicode-güvenli okuma/yazma (Windows TR yol desteği)
def imread_unicode(path: str, flags=cv2.IMREAD_GRAYSCALE):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)

def imwrite_unicode(path: str, img: np.ndarray, ext: str = ".png") -> bool:
    ensure_dir(os.path.dirname(path))
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True

# ============================== Önişleme ==============================
def _deskew(gray: np.ndarray, max_abs_angle: float = AIDIS_DESKEW_MAX) -> np.ndarray:
    H, W = gray.shape
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.size < 50:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) > max_abs_angle:
        return gray
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _illum_normalize(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    x = clahe.apply(gray)
    return cv2.GaussianBlur(x, (3, 3), 0)

def _binarize_adaptive(gray: np.ndarray) -> np.ndarray:
    blk = AIDIS_BLOCK if AIDIS_BLOCK % 2 == 1 else AIDIS_BLOCK + 1
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, AIDIS_C
    )
    # küçük gürültüyü temizle (açma); dilate yok → harfler yapışmasın
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return bw

def _equalize_resize_from_gray(gray: np.ndarray, size=None) -> np.ndarray:
    size = size or (AIDIS_IMG_SIZE, AIDIS_IMG_SIZE)
    im = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    im = cv2.equalizeHist(im)
    return im

# ============================== Satır & Harf Segmentasyonu ==============================
def _find_text_lines(bw: np.ndarray) -> list[tuple[int, int]]:
    H, W = bw.shape
    proj = (bw > 0).sum(axis=1).astype(np.int32)
    if proj.max() == 0:
        return []
    thr = max(3, int(0.02 * W))
    lines, in_run, y0 = [], False, 0
    for y, v in enumerate(proj):
        if not in_run and v >= thr:
            in_run, y0 = True, y
        elif in_run and v < thr:
            in_run = False
            y1 = y
            if y1 - y0 >= 5:
                lines.append((y0, y1))
    if in_run:
        lines.append((y0, H))
    return lines

def _split_wide_component(line_bw: np.ndarray, x0: int, y0: int, x: int, y: int, w: int, h: int, med_w: float):
    crops = []
    roi = line_bw[y:y+h, x:x+w]  # INV: harfler beyaz
    col = (roi > 0).sum(axis=0)
    maxv = col.max() if col.size else 0
    if maxv == 0:
        return [(x + x0, y + y0, w, h)]
    gap_thr = max(1, int(AIDIS_MIN_GAP_FR * roi.shape[0]))
    zero_cols = np.where(col <= gap_thr)[0]
    if zero_cols.size == 0:
        return [(x + x0, y + y0, w, h)]
    splits, start = [], zero_cols[0]
    for i in range(1, len(zero_cols)):
        if zero_cols[i] != zero_cols[i-1] + 1:
            splits.append((start, zero_cols[i-1])); start = zero_cols[i]
    splits.append((start, zero_cols[-1]))
    cuts = [int((a+b)//2) for (a,b) in splits if (b-a+1) >= 2]
    if not cuts:
        return [(x + x0, y + y0, w, h)]
    xs = [0] + cuts + [w]
    for i in range(len(xs)-1):
        xx0, xx1 = xs[i], xs[i+1]
        ww = xx1 - xx0
        if ww < 3:  # aşırı ince parçayı at
            continue
        crops.append((x + xx0 + x0, y + y0, ww, h))
    if not crops:
        crops = [(x + x0, y + y0, w, h)]
    return crops

def _segment_letters(gray: np.ndarray) -> list[tuple[int,int,int,int]]:
    H, W = gray.shape
    g = _deskew(gray)
    g = _illum_normalize(g)
    bw = _binarize_adaptive(g)

    lines = _find_text_lines(bw) or [(0, H)]
    boxes = []
    min_area = max(20, int(AIDIS_MIN_AREA * H * W))

    for (y0, y1) in lines:
        line_bw = bw[y0:y1, :]
        cnts, _ = cv2.findContours(line_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < min_area or h < 5 or w < 3:
                continue
            if w > AIDIS_MAX_W_RATIO * W and h > 0.5*(y1 - y0):
                continue
            raw.append((x, y, w, h))
        if not raw:
            continue

        # i-noktası, minik leke vb. ele: medyan yüksekliğin %40'ından küçükler
        med_h_all = np.median([h for (_,_,_,h) in raw]) if raw else 0
        min_h = int(max(5, AIDIS_MIN_HEIGHT_FR * med_h_all))
        raw = [(x,y,w,h) for (x,y,w,h) in raw if h >= min_h]
        if not raw:
            continue

        raw.sort(key=lambda b: b[0])
        med_w = np.median([w for (_,_,w,_) in raw]) if raw else 10
        for (x, y, w, h) in raw:
            if w >= AIDIS_SPLIT_W_MUL * med_w and w > 10:
                pieces = _split_wide_component(line_bw, 0, y0, x, y, w, h, med_w)
                boxes.extend(pieces)
            else:
                boxes.append((x, y + y0, w, h))

    if boxes:
        hs = [h for (_,_,_,h) in boxes]
        med_h = int(np.median(hs)) if hs else 20
        line_h = max(12, int(0.7 * med_h))
        def key(b):
            x,y,w,h = b
            row = y // max(1, line_h)
            return (row, x)
        boxes.sort(key=key)
    return boxes

# ============================== Tahmin & Anotasyon ==============================
def _predict_letter28(model, letter28: np.ndarray) -> float:
    X = (letter28.astype(np.float32) / 255.0)[None, ..., None]
    p = float(model.predict(X, verbose=0).ravel()[0])  # p_normal
    if not AIDIS_USE_INVERT_TTA:
        return p
    inv = 255 - letter28
    Xi = (inv.astype(np.float32) / 255.0)[None, ..., None]
    pi = float(model.predict(Xi, verbose=0).ravel()[0])
    return max(p, pi)

def process_image_and_predict(uploaded_image_path: str):
    """
    Dönüş:
      {
        "annotated_path": str,
        "dys": int,
        "normal": int,
        "total": int,
        "percent": float,
        "boxes": list[(x,y,w,h,label,prob_normal,margin)]
      }
    """
    model = get_model()
    img = imread_unicode(uploaded_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {uploaded_image_path}")

    boxes = _segment_letters(img)

    dys_count, normal_count = 0, 0
    ann = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out_boxes = []

    for (x, y, w, h) in boxes:
        crop_gray = img[y:y+h, x:x+w]
        letter28 = _equalize_resize_from_gray(crop_gray, size=(AIDIS_IMG_SIZE, AIDIS_IMG_SIZE))
        p_n = _predict_letter28(model, letter28)      # P(normal)
        p_d = 1.0 - p_n
        margin = p_d - p_n

        # --- Binary karar (uncertain YOK) ---
        if (p_d >= AIDIS_DYS_MIN) and (margin >= AIDIS_MARGIN_MIN):
            pred_label = "dyslexic"
            color = (0, 0, 255); dys_count += 1
        else:
            pred_label = "normal"
            color = (0, 200, 0); normal_count += 1

        cv2.rectangle(ann, (x, y), (x + w, y + h), color, 2)
        cv2.putText(ann, pred_label, (x, max(10, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        out_boxes.append((x, y, w, h, pred_label, float(p_n), float(margin)))

    total = dys_count + normal_count
    percent = (dys_count / total * 100.0) if total > 0 else 0.0

    out_dir = os.path.join(settings.MEDIA_ROOT, "annotated")
    ensure_dir(out_dir)
    annotated_name = os.path.splitext(os.path.basename(uploaded_image_path))[0] + "_annot.png"
    annotated_path = os.path.join(out_dir, annotated_name)
    if not imwrite_unicode(annotated_path, ann, ext=".png"):
        raise IOError(f"Annot görseli yazılamadı: {annotated_path}")

    return {
        "annotated_path": annotated_path,
        "dys": dys_count,
        "normal": normal_count,
        "total": total,
        "percent": round(percent, 2),
        "boxes": out_boxes,
    }
