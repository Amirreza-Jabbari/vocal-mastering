from django import forms
from .models import VocalMastering

class VocalUploadForm(forms.ModelForm):
    class Meta:
        model = VocalMastering
        fields = ['original_audio']
        widgets = {
            'original_audio': forms.FileInput(attrs={'accept': 'audio/*'})
        }