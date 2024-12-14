from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_vocal, name='upload_vocal'),
    path('job/<uuid:job_id>/status/', views.job_status, name='job_status'),
    path('job/<uuid:job_id>/download/', views.download_mastered_audio, name='download_mastered_audio'),
]