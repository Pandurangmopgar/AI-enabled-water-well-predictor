from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load the model and scaler
clf_multi_param = joblib.load('path/to/your/random_forest_model.pkl')
scaler = joblib.load('path/to/your/scaler.pkl')

# Load the dataset
df = pd.read_csv('path/to/your/data_20221_cleaned.csv')

@csrf_exempt  # Disable CSRF token for this view
@require_http_methods(["POST"])  # Only allow POST requests
def coordinates(request):
    # Assuming the frontend sends a JSON with 'lat' and 'lng'
    data = json.loads(request.body.decode('utf-8'))
    lat = data['lat']
    lng = data['lng']

    # The rest of the processing is as per your Flask app
    # ...

    # After processing and getting the prediction
    prediction_label = # ... result from your model ...

    # Return the prediction as JSON
    return JsonResponse({"message": f"Water quality is {prediction_label}"})
