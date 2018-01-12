#!/usr/bin/env python3.5.2
# -*- coding: utf-8

from . import views
from django.urls import path

urlpatterns = [
    path('analysis/', views.picAnalysis),
    path('save/',views.savePic),
    path('train/',views.trainPic),

]