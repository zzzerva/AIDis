# core/views.py
import os
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import FileResponse
from .forms import UploadForm
from .models import Upload
from .utils import process_image_and_predict
from django.core.files import File 
from reportlab.lib import colors

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors  # <-- RENKLER için

def _register_tr_fonts():
    fonts_dir = os.path.join(settings.BASE_DIR, "assets", "fonts")
    normal_ttf = os.path.join(fonts_dir, "DEJAVUSANS.TTF")
    bold_ttf   = os.path.join(fonts_dir, "DEJAVUSANS-BOLD.TTF")

    BASE_FONT = "Helvetica"
    BASE_FONT_BOLD = "Helvetica-Bold"
    try:
        if os.path.exists(normal_ttf):
            pdfmetrics.registerFont(TTFont("TRFont", normal_ttf))
            BASE_FONT = "TRFont"
        if os.path.exists(bold_ttf):
            pdfmetrics.registerFont(TTFont("TRFont-Bold", bold_ttf))
            BASE_FONT_BOLD = "TRFont-Bold"
    except Exception:
        pass
    return BASE_FONT, BASE_FONT_BOLD


def _draw_kv_table(
    c,
    x,
    y_start,
    col1_w,
    col2_w,
    rows,
    row_h=22,
    font="Helvetica",
    font_bold="Helvetica-Bold",
    palette=None,
):
    """
    Profesyonel görünümlü iki sütunlu tablo:
    - Sol sütun (alan adları): koyu gri (göz yormaz)
    - Sağ sütun (değerler): siyah
    - İnce grid çizgileri + zebra satır arka planı
    """
    if palette is None:
        palette = {
            "label": colors.HexColor("#374151"),  # slate-700
            "value": colors.black,
            "grid":  colors.HexColor("#D1D5DB"),  # gray-300
            "rowbg": colors.HexColor("#F9FAFB"),  # gray-50
        }

    y = y_start
    c.setLineWidth(0.5)
    for i, (label, value) in enumerate(rows):
        # Zebra satır arka planı (tam satır genişliği)
        if i % 2 == 0:
            c.setFillColor(palette["rowbg"])
            c.rect(x, y - row_h, col1_w + col2_w, row_h, fill=1, stroke=0)

        # Hücre kenar çizgileri (ince gri)
        c.setStrokeColor(palette["grid"])
        c.rect(x, y - row_h, col1_w, row_h, fill=0, stroke=1)
        c.rect(x + col1_w, y - row_h, col2_w, row_h, fill=0, stroke=1)

        # Sol hücre: alan adı (koyu gri, yarı kalın)
        c.setFont(font_bold, 11)
        c.setFillColor(palette["label"])
        c.drawString(x + 8, y - row_h + 6, str(label))

        # Sağ hücre: değer (siyah)
        c.setFont(font, 11)
        c.setFillColor(palette["value"])
        c.drawString(x + col1_w + 8, y - row_h + 6, str(value))

        y -= row_h

    return y



def generate_pdf(upload_obj, result):
    BASE_FONT, BASE_FONT_BOLD = _register_tr_fonts()

    pdf_dir = os.path.join(settings.MEDIA_ROOT, "uploads", "annotated", "reports")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"report_{upload_obj.id}.pdf")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    W, H = A4
    M = 72  # kenar boşluğu

    # Başlık
    c.setFont(BASE_FONT_BOLD, 18)
    c.setFillColor(colors.black)
    c.drawString(M, H - M, "AIDis: Disleksi Analiz Raporu")

    # Tablo içerikleri
    total = upload_obj.dyslexic_count + upload_obj.normal_count
    rows = [
        ("Ad",            upload_obj.first_name),
        ("Soyad",         upload_obj.last_name),
        ("Yaş",           upload_obj.age),
        ("Toplam Harf",   total),
        ("Normal Harf",   upload_obj.normal_count),
        ("Disleksi Harf", upload_obj.dyslexic_count),
        ("Dislektik Harf Oranı", f"%{upload_obj.dyslexic_percent}"),
    ]

    # Tablo ölçüleri
    col1_w = 160
    col2_w = (W - 2 * M) - col1_w
    y_start = H - M - 28

    # Yumuşak palet (göz yormayan)
    palette = {
        "label": colors.HexColor("#374151"),  # koyu gri (slate-700)
        "value": colors.black,
        "grid":  colors.HexColor("#D1D5DB"),  # açık gri grid
        "rowbg": colors.HexColor("#F9FAFB"),  # zebra arka plan
    }

    y = _draw_kv_table(
        c, x=M, y_start=y_start, col1_w=col1_w, col2_w=col2_w,
        rows=rows, row_h=24, font=BASE_FONT, font_bold=BASE_FONT_BOLD,
        palette=palette
    )
    y -= 14

    # Yönlendirme mesajı: semantik olarak "uyarı" renkli kalsın ama daha "soft" kırmızı
    warn_color = colors.HexColor("#B91C1C")  # red-700 (daha koyu, daha az parlak)
    msg = ("Özel eğitim uzmanı veya psikiyatriye yönlendirilmelisiniz."
           if upload_obj.dyslexic_percent > 45 else
           "Normal sınırlar içinde.")
    c.setFont(BASE_FONT_BOLD, 12)
    c.setFillColor(warn_color if upload_obj.dyslexic_percent > 50 else colors.HexColor("#166534"))  # else: yeşil-700
    c.drawString(M, y, msg)
    y -= 18

    # Annot görseli (eski mantık)
    img_abs = None
    if upload_obj.annotated_image:
        p = os.path.join(settings.MEDIA_ROOT, upload_obj.annotated_image.name)
        if os.path.exists(p):
            img_abs = p

    if img_abs:
        try:
            ir = ImageReader(img_abs)
            iw, ih = ir.getSize()
            max_w, max_h = (W - 2 * M), 360
            scale = min(max_w / iw, max_h / ih, 1.0)
            dw, dh = iw * scale, ih * scale
            x_img, y_img = M, max(M, y - dh)
            c.setFillColor(colors.black)
            c.drawImage(ir, x_img, y_img, width=dw, height=dh, preserveAspectRatio=True, mask='auto')
            y = y_img - 12
        except Exception as e:
            c.setFont(BASE_FONT, 10)
            c.setFillColor(colors.black)
            c.drawString(M, y, f"Annot görseli eklenemedi: {e}")
            y -= 14

    c.showPage()
    c.save()

    # FileField'e yaz
    with open(pdf_path, "rb") as f:
        upload_obj.pdf_report.save(os.path.basename(pdf_path), File(f), save=True)

    return pdf_path


from django.contrib import messages
from django.db import transaction

def upload_view(request):
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        try:
            if form.is_valid():
                with transaction.atomic():
                    upload = form.save()

                    # Ana işlem
                    result = process_image_and_predict(upload.image.path)

                    # Annoted görseli ve metrikleri kaydet
                    rel_annotated = os.path.relpath(result["annotated_path"], settings.MEDIA_ROOT)
                    upload.annotated_image.name = rel_annotated
                    upload.dyslexic_count = result["dys"]
                    upload.normal_count = result["normal"]
                    upload.dyslexic_percent = result["percent"]
                    upload.save()

                    # PDF oluştur
                    generate_pdf(upload, result)

                    return redirect("result", pk=upload.pk)
            else:
                messages.error(request, "Form geçersiz. Lütfen alanları kontrol edin.")
        except Exception as e:
            try:
                if 'upload' in locals() and upload and upload.pk:
                    upload.delete()
            except Exception:
                pass
            import traceback
            traceback.print_exc()
            messages.error(request, f"Analiz sırasında hata: {type(e).__name__}: {e!s}")
    else:
        form = UploadForm()

    return render(request, "upload.html", {"form": form})


def result_view(request, pk):
    upload = Upload.objects.get(pk=pk)
    return render(request, "result.html", {"upload": upload})


def download_pdf(request, pk):
    upload = Upload.objects.get(pk=pk)
    if not upload.pdf_report:
        return redirect("result", pk=pk)
    return FileResponse(open(upload.pdf_report.path, "rb"), as_attachment=True)
