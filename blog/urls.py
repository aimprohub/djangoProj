from django.urls import path 
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name = 'blog-home'),
    path('about/', views.about, name = 'blog-about'),
    path('prediction/', views.prediction, name = 'blog-prediction'),
    path('upload/', views.upload, name = 'blog-upload'),
    # path('name/', views.name, name = 'blog-name'),

    # path('prediction/', views.trial, name='trial'),
    path('prediction/trial/', views.trial, name='trial'),
    path('prediction/trial/name/', views.name, name='name'),
    path('upload/read/', views.read, name='read'),
    path('upload/read/name2/', views.name2, name='read2'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)