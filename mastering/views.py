import logging
from django.shortcuts import render, redirect
from django.http import JsonResponse, FileResponse, Http404
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.core.exceptions import ValidationError
from .models import VocalMastering
from .forms import VocalUploadForm
from .processors import SmarterVocalMasteringProcessor

logger = logging.getLogger(__name__)

@require_http_methods(["GET", "POST"])
def upload_vocal(request):
    if request.method == 'POST':
        form = VocalUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            audio_file = request.FILES.get('original_audio')
            _validate_audio_file(audio_file)
            job = VocalMastering.objects.create(original_audio=audio_file)
            _process_vocal_async(job)
            return redirect('job_status', job_id=job.id)
        
        messages.error(request, "Invalid form submission")
    
    else:
        form = VocalUploadForm()
    
    return render(request, 'mastering/upload.html', {'form': form})

def _validate_audio_file(audio_file):
    if not audio_file:
        raise ValidationError("No audio file uploaded")
    
    max_size = 50 * 1024 * 1024
    if audio_file.size > max_size:
        raise ValidationError(f"File too large. Maximum size is {max_size/1024/1024}MB")
    
    allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/x-wav', 'audio/x-m4a']
    if audio_file.content_type not in allowed_types:
        raise ValidationError("Invalid file type. Supported: WAV, MP3, M4A")

def _process_vocal_async(job):
    from threading import Thread
    
    def process_task():
        try:
            processor = SmarterVocalMasteringProcessor()
            processed_path = processor.process_vocal(job.original_audio.path)
            if processed_path:
                with open(processed_path, 'rb') as f:
                    job.mastered_audio.save(f'mastered_{job.id}.wav', f)
                job.save()
        except Exception as e:
            logger.error(f"Processing error for job {job.id}: {e}")
    
    Thread(target=process_task, daemon=True).start()

def job_status(request, job_id):
    try:
        job = VocalMastering.objects.get(id=job_id)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'status': 'COMPLETED' if job.mastered_audio else 'PROCESSING',
                'mastered_audio_url': job.mastered_audio.url if job.mastered_audio else None
            })
        return render(request, 'mastering/job_status.html', {'job': job})
    
    except VocalMastering.DoesNotExist:
        messages.error(request, "Job not found")
        return redirect('upload_vocal')

def download_mastered_audio(request, job_id):
    try:
        job = VocalMastering.objects.get(id=job_id)
        if not job.mastered_audio:
            raise Http404("Mastered audio file not found")
        
        response = FileResponse(job.mastered_audio.open('rb'), as_attachment=True, filename=f'mastered_{job.id}.wav')
        response['Content-Type'] = 'audio/wav'
        return response
    
    except VocalMastering.DoesNotExist:
        raise Http404("Job not found")