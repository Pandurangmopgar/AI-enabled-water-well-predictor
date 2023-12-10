from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from flask_cors import CORS  # Import CORS
import logging
# app = Flask(__name__)
# CORS(app)  # Enable CORS on your Flask app
# from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # CORS applied to /coordinates route for all origins
logging.basicConfig(level=logging.INFO)
# Load the model and scaler
clf_multi_param = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the dataset
df = pd.read_csv('data_20221_cleaned.csv')

# Convert LATITUDE and LONGITUDE to numeric and drop NaN values
df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
df.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)

# Fit the NearestNeighbors model
coordinates = df[['LATITUDE', 'LONGITUDE']].values
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(coordinates)
@app.route('/')
def index():
    return 'Welcome to the Water Quality Prediction API!'


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate and extract lat and lng from request
        data = request.json
        lat = float(data.get('latitude'))  # Use .get to avoid KeyError
        lng = float(data.get('longitude'))

        # Find the nearest coordinates if exact match not found
        distances, indices = neigh.kneighbors(np.array([[lat, lng]]))
        nearest_index = indices[0][0]
        query_result = df.iloc[[nearest_index]]

        # Prepare the features for prediction
        features = query_result.drop(columns=['Well ID', 'S.No', 'STATE', 'DISTRICT', 'BLOCK', 'LOCATION', 'LATITUDE', 'LONGITUDE', 'Year', 'PO4', 'SiO2', 'TDS', 'U(ppb)'])
        features_scaled = scaler.transform(features)

        # Make a prediction
        prediction = clf_multi_param.predict(features_scaled)

        # Convert numerical prediction back to label
        label_map = {0: 'Good', 1: 'Poor'}
        prediction_label = label_map[prediction[0]]
        app.logger.info(f'Predicted label: {prediction_label} for coordinates lat: {lat}, lng: {lng}')
        print(f'Predicted label: {prediction_label} for coordinates lat: {lat}, lng: {lng}')  # This will display in the console

         
        # Return the prediction in a JSON response
        return jsonify({
            "prediction":prediction_label
        })

    except Exception as e:
        app.logger.error(f'Error: {e}')
        return jsonify({"error": "An error occurred during processing"}), 500

if __name__ == '__main__':
    
#  coordinates()    
 app.run(debug=True)
