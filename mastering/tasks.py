from celery import shared_task
from django.core.files import File
from .models import VocalMastering
from .processors import SmarterVocalMasteringProcessor
from django.utils import timezone
import os

@shared_task
def process_vocal_track(job_id):
    try:
        job = VocalMastering.objects.get(id=job_id)
        processor = SmarterVocalMasteringProcessor()
        processed_file_path = processor.process_vocal(job.original_audio.path)

        if processed_file_path:
            with open(processed_file_path, 'rb') as f:
                job.mastered_audio.save(os.path.basename(processed_file_path), File(f))
            job.status = 'completed'
            job.completed_at = timezone.now()
        else:
            job.status = 'failed'
            job .error_message = "Processing failed"
        
        job.save()
        
    except Exception as e:
        job.status = 'failed'
        job.error_message = str(e)
        job.save()