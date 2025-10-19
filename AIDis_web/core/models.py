from django.db import models



class Upload(models.Model):
    first_name = models.CharField('Ad', max_length=100)
    last_name = models.CharField('Soyad', max_length=100)
    age = models.PositiveIntegerField('Yaş')
    image = models.ImageField(upload_to='uploads/')
    annotated_image = models.ImageField(upload_to='annotated/', null=True, blank=True)


    dyslexic_count = models.IntegerField(default=0)
    normal_count = models.IntegerField(default=0)
    dyslexic_percent = models.FloatField(default=0.0)


    pdf_report = models.FileField(upload_to='reports/', null=True, blank=True)


    created_at = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return f"{self.first_name} {self.last_name} - {self.created_at.date()}"