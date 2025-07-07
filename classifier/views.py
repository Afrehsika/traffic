import os
import datetime
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from classifier.models import load_model, predict

# Load the model once globally
model = load_model()

# Global variable to store the latest prediction and location
latest_prediction = {
    "result": None,
    "timestamp": None,
    "lat": None,
    "lon": None
}


# Camera device: index page with webcam
def index(request):
    return render(request, 'index.html')


# Receiver device: map and alerts
def receiver_view(request):
    return render(request, 'receiver.html')


# POST from camera: receives image + GPS, returns prediction
@csrf_exempt
def predict_webcam(request):
    global latest_prediction

    if request.method == 'POST' and request.FILES.get('image'):
        img = request.FILES['image']
        lat = request.POST.get('lat')
        lon = request.POST.get('lon')

        # Save uploaded image to static folder
        upload_dir = os.path.join('classifier', 'static')
        os.makedirs(upload_dir, exist_ok=True)
        image_path = os.path.join(upload_dir, 'webcam.jpg')
        with open(image_path, 'wb+') as f:
            for chunk in img.chunks():
                f.write(chunk)

        # Make prediction
        result = predict(image_path, model)

        # Save latest result and location
        latest_prediction = {
            "result": result,
            "timestamp": datetime.datetime.now().isoformat(),
            "lat": lat,
            "lon": lon
        }

        return JsonResponse({'result': result})

    return JsonResponse({'error': 'Invalid request'}, status=400)


# GET from receiver: returns latest prediction + GPS
def get_latest_prediction(request):
    return JsonResponse(latest_prediction)
