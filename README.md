# AIDis: Disleksi Analiz Sistemi

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://tensorflow.org)
[![Django](https://img.shields.io/badge/Django-4.2+-green.svg)](https://djangoproject.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AIDis, yapay zeka teknolojileri kullanarak disleksi analizi yapan kapsamlı bir sistemdir. Sistem, hem makine öğrenmesi pipeline'ı hem de web arayüzü içerir.

## 🎯 Özellikler

- **CNN Tabanlı Analiz**: 28x28 piksel harf görüntülerini sınıflandıran derin öğrenme modeli
- **Web Arayüzü**: Django tabanlı kullanıcı dostu web uygulaması
- **Otomatik Segmentasyon**: Kelime ve cümle görüntülerini harflere bölen akıllı algoritma
- **PDF Raporları**: Analiz sonuçlarını içeren detaylı PDF raporları
- **Gerçek Zamanlı Tahmin**: Yüklenen görüntülerin anında analizi
- **Türkçe Dil Desteği**: Tam Türkçe arayüz ve raporlama

## 📁 Proje Yapısı

```
AIDisMihri/
├── AIDis_model/               # ML Pipeline & Web Uygulaması
│   ├── src/                   # ML Kaynak kodlar
│   │   ├── evaluate.py        # Ana değerlendirme scripti
│   │   ├── evaluate_dataset.py # Dataset değerlendirme
│   │   ├── model.py           # CNN model tanımı
│   │   ├── prepare_data.py    # Veri hazırlama
│   │   └── train.py           # Model eğitimi
│   ├── aidis_project/         # Django proje ayarları
│   ├── core/                  # Django ana uygulama
│   │   ├── models.py          # Veritabanı modelleri
│   │   ├── views.py           # Görünümler
│   │   ├── utils.py           # Yardımcı fonksiyonlar
│   │   └── forms.py           # Formlar
│   ├── templates/             # HTML şablonları
│   ├── static/                # Statik dosyalar
│   ├── media/                 # Yüklenen dosyalar
│   ├── config.py              # ML Konfigürasyon
│   ├── utils.py               # ML Yardımcı fonksiyonlar
│   ├── data/                  # Veri dosyaları
│   ├── test_images/           # Test görselleri
│   ├── manage.py              # Django yönetim scripti
│   └── requirements.txt       # Python bağımlılıkları
├── saved_models/              # Eğitilmiş model dosyaları
├── .gitignore                 # Git ignore kuralları
├── .pre-commit-config.yaml    # Pre-commit hooks
└── README.md                  # Bu dosya
```

## 🚀 Kurulum

### Gereksinimler

- Python 3.8+
- TensorFlow 2.12+
- OpenCV
- Django 4.2+
- Diğer bağımlılıklar (requirements.txt dosyalarında)

### Kurulum

```bash
# Projeyi klonlayın
git clone https://github.com/yourusername/AIDisMihri.git
cd AIDisMihri

# Sanal ortam oluşturun
cd AIDis_model
python -m venv env

# Sanal ortamı aktifleştirin
# Windows:
env\Scripts\activate
# Linux/Mac:
source env/bin/activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Veritabanını oluşturun
python manage.py migrate

# Statik dosyaları toplayın
python manage.py collectstatic

# Web sunucusunu başlatın
python manage.py runserver
```

## 🔧 Kullanım

### ML Pipeline

#### Model Eğitimi
```bash
cd AIDis_model
python src/train.py
```

#### Veri Hazırlama
```bash
python src/prepare_data.py
```

#### Model Değerlendirme
```bash
python src/evaluate.py
```

#### Dataset Değerlendirme
```bash
python src/evaluate_dataset.py
```

### Web Uygulaması

1. Web sunucusunu başlatın:
```bash
cd AIDis_model
python manage.py runserver
```

2. Tarayıcınızda `http://127.0.0.1:8000` adresine gidin

3. Görsel yükleyin ve analiz sonuçlarını görün

## ⚙️ Konfigürasyon

### Konfigürasyon

`AIDis_model/config.py` dosyasından ML ayarlarını değiştirebilirsiniz:

```python
# Model ayarları
"IMG_SIZE": (28, 28),
"THRESHOLD": 0.25,

# Dosya yolları
"MODEL_PATH": ROOT / "final_model.keras",
"TEST_DIR": ROOT / "test_images",
```

`AIDis_model/core/utils.py` dosyasından web uygulaması ayarlarını değiştirebilirsiniz:

```python
# Model ayarları
AIDIS_IMG_SIZE = 28
AIDIS_DYS_MIN = 0.80
AIDIS_MARGIN_MIN = 0.20

# Segmentasyon ayarları
AIDIS_MIN_AREA = 0.0014
AIDIS_MIN_HEIGHT_FR = 0.40
```

## 📊 Model Performansı

Model, aşağıdaki metriklerle değerlendirilir:

- **Accuracy**: Genel doğruluk
- **Sensitivity**: Normal harfleri doğru tanıma oranı
- **Specificity**: Disleksi harfleri doğru tanıma oranı
- **AUC-ROC**: ROC eğrisi altındaki alan
- **Macro-F1**: Makro F1 skoru

## 🧪 Test

### Test
```bash
cd AIDis_model

# ML Pipeline Testi
python src/evaluate.py

# Web Uygulaması Testi
python manage.py test
```

## 📈 Veri Formatı

### Giriş Verileri

- **Test Görselleri**: `AIDis_model/test_images/normal/` ve `AIDis_model/test_images/dyslexic/` klasörlerinde
- **Model Dosyaları**: `.keras` veya `.h5` formatında

### Çıkış Verileri

- **Tahmin Sonuçları**: CSV formatında
- **Annotasyonlu Görseller**: PNG formatında
- **PDF Raporları**: Analiz sonuçları

## 🔍 Sorun Giderme

### Yaygın Sorunlar

1. **Model dosyası bulunamadı**
   - Model dosyalarının doğru konumda olduğundan emin olun
   - `config.py` dosyasındaki yolları kontrol edin

2. **Import hatası**
   - Sanal ortamın aktif olduğundan emin olun
   - Bağımlılıkların yüklendiğini kontrol edin

3. **Django hatası**
   - Veritabanı migrasyonlarını çalıştırın
   - Statik dosyaları toplayın: `python manage.py collectstatic`

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 📞 İletişim

- **Proje Sahibi**: [GitHub Profili](https://github.com/yourusername)
- **E-posta**: your.email@example.com
- **GitHub Issues**: [Issues Sayfası](https://github.com/yourusername/AIDisMihri/issues)

## 🙏 Teşekkürler

- TensorFlow ekibine
- OpenCV ekibine
- Django ekibine
- Tüm katkıda bulunanlara

## 📚 Referanslar

- [TensorFlow Documentation](https://tensorflow.org/docs)
- [Django Documentation](https://docs.djangoproject.com)
- [OpenCV Documentation](https://docs.opencv.org)

---

**AIDis** - Disleksi analizinde yapay zeka ile fark yaratın! 🧠✨