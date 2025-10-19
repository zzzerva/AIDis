"""
AIDis Makale Parse Konfigürasyonu
"""

import os
from pathlib import Path
from typing import Dict, Any

# Proje kök dizini (bu dosyanın bulunduğu klasör)
ROOT = Path(__file__).parent.resolve()

# Varsayılan ayarlar
DEFAULT_CONFIG = {
    # Proje yolları
    "ROOT": ROOT,
    # Model ayarları
    "IMG_SIZE": (28, 28),
    "BATCH_SIZE": 64,
    "THRESHOLD": 0.25,
    
    # Dosya yolları
ARTIFACTS_DIR = ROOT / "models"

PATHS = {
    "MODEL_PATH":     ARTIFACTS_DIR / "final_model.keras",
    "CLASS_IDX_PATH": ARTIFACTS_DIR / "class_indices.json",
    "TEST_DIR":       ARTIFACTS_DIR / "test_images",
    "CROPS_DIR":      ARTIFACTS_DIR / "letter_crops",
    "ANNOT_DIR":      ARTIFACTS_DIR / "letter_crops_annotated",
}
    
    # Çıktı dosyaları
    "CROPS_PRED_CSV": ROOT / "predictions_letter_crops.csv",
    "TEST_PRED_CSV": ROOT / "predictions_test_images.csv",
    
    # Log/Kayıt ayarları
    "SAVE_CSV": True,
    "PRINT_PER_SAMPLE": True,
    "SAVE_MISCLASSIFIED": True,
    "SAVE_DEBUG_28x28": False,
    "DRAW_ANNOTATED": True,
    
    # TTA ayarları
    "TTA_USE_STRICT": True,
    "TTA_USE_INVERT": False,
    "TTA_REDUCER": "max",
    
    # Kelime parser ayarları
    "LETTER_PARSER_ENABLED": True,
    
    # Sınıf haritası
    "CLASS_MAP": {"dyslexic": 0, "normal": 1}
}

def get_config() -> Dict[str, Any]:
    """Konfigürasyonu döndür"""
    config = DEFAULT_CONFIG.copy()
    
    # Ortam değişkenlerinden override
    if os.getenv("AIDIS_IMG_SIZE"):
        config["IMG_SIZE"] = tuple(map(int, os.getenv("AIDIS_IMG_SIZE").split(",")))
    
    if os.getenv("AIDIS_THRESHOLD"):
        config["THRESHOLD"] = float(os.getenv("AIDIS_THRESHOLD"))
    
    if os.getenv("AIDIS_MODEL_PATH"):
        config["MODEL_PATH"] = Path(os.getenv("AIDIS_MODEL_PATH"))
    
    return config

def get_model_path() -> Path:
    """Model dosyası yolunu döndür"""
    config = get_config()
    model_path = config["MODEL_PATH"]
    
    # Eğer .keras yoksa .h5'ı dene
    if not model_path.exists() and model_path.suffix == ".keras":
        h5_path = model_path.with_suffix(".h5")
        if h5_path.exists():
            return h5_path
    
    return model_path

def get_class_indices() -> Dict[str, int]:
    """Sınıf haritasını döndür"""
    config = get_config()
    class_idx_path = config["CLASS_IDX_PATH"]
    
    if class_idx_path.exists():
        import json
        with open(class_idx_path, "r") as f:
            return json.load(f)
    else:
        return config["CLASS_MAP"]
