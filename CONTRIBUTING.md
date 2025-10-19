# Katkıda Bulunma Rehberi

AIDis projesine katkıda bulunmak için bu rehberi takip edin.

## 🚀 Başlangıç

1. Projeyi fork edin
2. Yerel makinenizde klonlayın:
   ```bash
   git clone https://github.com/yourusername/AIDisMihri.git
   cd AIDisMihri
   ```
3. Upstream remote'u ekleyin:
   ```bash
   git remote add upstream https://github.com/originalowner/AIDisMihri.git
   ```

## 🔧 Geliştirme Ortamı

### Gereksinimler
- Python 3.8+
- Git
- Virtual environment

### Kurulum
```bash
# Proje için
cd AIDis_model
python -m venv env
source env/bin/activate  # Linux/Mac
# veya
env\Scripts\activate     # Windows
pip install -r requirements.txt
```

## 📝 Katkı Süreci

### 1. Issue Oluşturma
- Yeni özellik veya bug için issue oluşturun
- Mevcut issue'ları kontrol edin
- Açık ve net bir başlık kullanın

### 2. Branch Oluşturma
```bash
git checkout -b feature/your-feature-name
# veya
git checkout -b bugfix/your-bug-description
```

### 3. Kod Yazma
- Kod standartlarına uyun
- Türkçe yorum satırları kullanın
- Fonksiyon ve sınıflar için docstring yazın
- Test yazın (mümkünse)

### 4. Commit Mesajları
```
feat: yeni özellik eklendi
fix: bug düzeltildi
docs: dokümantasyon güncellendi
style: kod formatı düzeltildi
refactor: kod yeniden düzenlendi
test: test eklendi
chore: build/ci güncellendi
```

### 5. Push ve Pull Request
```bash
git push origin feature/your-feature-name
```

## 🎯 Kod Standartları

### Python
- PEP 8 standartlarına uyun
- Black formatter kullanın
- Type hints ekleyin
- Docstring'leri Türkçe yazın

### Django
- Django best practices'i takip edin
- Model'lerde verbose_name kullanın
- View'larda try-except kullanın
- Template'lerde semantic HTML kullanın

### Git
- Anlamlı commit mesajları yazın
- Küçük, odaklanmış commit'ler yapın
- Merge commit'lerden kaçının

## 🧪 Test

### Test
```bash
cd AIDis_model

# ML Pipeline Testi
python src/evaluate.py

# Web Uygulaması Testi
python manage.py test
```

## 📝 Dokümantasyon

- README.md'yi güncelleyin
- Yeni özellikler için dokümantasyon ekleyin
- Kod yorumlarını Türkçe yazın
- API dokümantasyonu ekleyin (gerekirse)

## 🐛 Bug Raporlama

Bug raporu oluştururken şunları ekleyin:
- Açık ve net başlık
- Bug'ın nasıl tekrarlanacağı
- Beklenen davranış
- Gerçek davranış
- Ekran görüntüleri (varsa)
- Sistem bilgileri

## ✨ Özellik İstekleri

Özellik isteği oluştururken:
- Açık ve net başlık
- Özelliğin ne işe yarayacağı
- Neden gerekli olduğu
- Alternatif çözümler (varsa)

## 📋 Pull Request Checklist

- [ ] Kod standartlarına uygun
- [ ] Test yazıldı (mümkünse)
- [ ] Dokümantasyon güncellendi
- [ ] Commit mesajları anlamlı
- [ ] Conflict yok
- [ ] CI/CD geçiyor

## 🤝 Code Review

- Yapıcı geri bildirim verin
- Kod kalitesine odaklanın
- Öğretici olun
- Zamanında yanıt verin

## 📞 İletişim

- GitHub Issues kullanın
- Discord/Slack (varsa)
- E-posta (gerekirse)

## 🙏 Teşekkürler

Katkıda bulunan herkese teşekkürler! 🎉
