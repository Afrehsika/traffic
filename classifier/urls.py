from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),                        # Camera interface
    path('receiver/', views.receiver_view, name='receiver'),    # Receiver interface
    path('predict/', views.predict_webcam, name='predict'),     # POST image + location
    path('latest/', views.get_latest_prediction, name='latest') # GET latest prediction
]
