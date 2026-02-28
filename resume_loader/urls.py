from django.urls import path
from resume_loader import views
urlpatterns=[
    path('upload/resume/',views.UploadResume),
    path('analyze/resume/',views.AnalyzeResume,name='analyzeResume')
]