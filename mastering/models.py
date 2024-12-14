import uuid
from django.db import models

class VocalMastering(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    original_audio = models.FileField(upload_to='vocals/')
    mastered_audio = models.FileField(upload_to='mastered/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Vocal Mastering - {self.id}"