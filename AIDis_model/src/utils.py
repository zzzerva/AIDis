"""
AIDis Makale Parse Yardımcı Fonksiyonlar
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config

def ensure_dir(path: str) -> None:
    """Dizini oluştur (yoksa)"""
    os.makedirs(path, exist_ok=True)

def imread_unicode(path: str, flags: int = cv2.IMREAD_GRAYSCALE) -> Optional[np.ndarray]:
    """
    Windows'ta non-ASCII (ör. 'Masaüstü') içeren yollarda güvenli okuma
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)

def imwrite_unicode(path: str, img: np.ndarray, ext: str = ".png") -> bool:
    """
    Windows'ta non-ASCII içeren yollarda güvenli yazma
    """
    ensure_dir(os.path.dirname(path))
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True

def equalize_resize_from_gray(gray: np.ndarray, size: Tuple[int, int] = (28, 28)) -> np.ndarray:
    """Eğitimle aynı preprocessing: resize + histogram eşitleme"""
    im = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    im = cv2.equalizeHist(im)
    return im

def preprocess_image(image_path: str, target_size: Tuple[int, int] = (28, 28)) -> Optional[np.ndarray]:
    """Görüntüyü tam preprocessing pipeline ile işle"""
    # Görüntüyü oku
    img = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Resize ve histogram eşitleme
    img = equalize_resize_from_gray(img, target_size)
    return img

def get_image_info(img: np.ndarray) -> dict:
    """Görüntü hakkında bilgi döndür"""
    if img is None:
        return {"error": "Görüntü None"}
    
    return {
        "shape": img.shape,
        "dtype": str(img.dtype),
        "min_value": float(img.min()),
        "max_value": float(img.max()),
        "mean_value": float(img.mean()),
        "channels": img.shape[2] if len(img.shape) == 3 else 1
    }

def sort_boxes_reading_order(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Bounding box'ları okuma sırasına göre sırala"""
    if not boxes:
        return []
    
    # Ortalama yüksekliği hesapla
    heights = [h for (_, _, _, h) in boxes]
    med_h = int(np.median(heights)) if heights else 20
    line_h = max(12, int(0.6 * med_h))
    
    def key(box):
        x, y, w, h = box
        row = y // line_h
        return (row, x)
    
    return sorted(boxes, key=key)

def segment_letters(gray_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Gri kelime/cümle görüntüsünden harfleri segment et"""
    H, W = gray_img.shape
    
    # Histogram eşitleme
    eq = cv2.equalizeHist(gray_img)
    blur = cv2.GaussianBlur(eq, (3, 3), 0)
    
    # Binarizasyon (harfler beyaz olacak şekilde)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morfolojik işlemler
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    bw = cv2.dilate(bw, np.ones((2, 2), np.uint8), iterations=1)
    
    # Konturları bul
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    min_area = max(20, int(0.0015 * H * W))
    
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        
        # Filtreleme
        if area < min_area or h < 5 or w < 3:
            continue
        if h > 0.9 * H and w < 0.1 * W:
            continue  # Dikey büyük parazit
        
        boxes.append((x, y, w, h))
    
    # Okuma sırasına göre sırala
    boxes = sort_boxes_reading_order(boxes)
    return boxes