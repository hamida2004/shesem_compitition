from django.urls import path
from . import views

urlpatterns = [
    path("recommendations/crop/", views.crop_recommendation, name="crop_recommendation"), path("recommendations/general/", views.general_recommendation, name="general_recommendation"),]