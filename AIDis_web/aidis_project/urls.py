from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from core import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.upload_view, name="upload"),
    path("result/<int:pk>/", views.result_view, name="result"),
    path("download/<int:pk>/", views.download_pdf, name="download_pdf"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
