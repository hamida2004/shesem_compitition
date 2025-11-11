from django.urls import path
from . import views

urlpatterns = [
    path("crop", views.crop_recommendation, name="crop_recommendation"),
    path("general", views.general_recommendation, name="general_recommendation"),
    path('next', views.next_crop),  
    ]