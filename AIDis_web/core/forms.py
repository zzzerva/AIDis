from django import forms
from .models import Upload


class UploadForm(forms.ModelForm):
    class Meta:
        model = Upload
        fields = ['first_name','last_name','age','image']
        widgets = {
        'first_name': forms.TextInput(attrs={'placeholder': 'Adınız'}),
        'last_name': forms.TextInput(attrs={'placeholder': 'Soyadınız'}),
        'age': forms.NumberInput(attrs={'min': 1}),
        }