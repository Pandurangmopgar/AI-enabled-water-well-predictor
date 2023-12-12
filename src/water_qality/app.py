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
# clf_multi_param = joblib.load('random_forest_model.pkl')
# scaler = joblib.load('scaler.pkl')
path="E:\\GWL\\src\\water_qality\\finalized_model.pkl"
model=joblib.load(path)
# Load the dataset
df = pd.read_csv('water_qality\\adjusted_water_quality_data_v2.csv')

# Convert LATITUDE and LONGITUDE to numeric and drop NaN values
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Fit the NearestNeighbors model
coordinates = df[['Latitude', 'Longitude']].values
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
        #  q   uery_result

        features=query_result[['pH', 'Cl', 'NO3', 'TH',  'Ca', 'Mg','TDS']]
# ['pH', 'Cl', 'NO3', 'TH',  'Ca', 'Mg','TDS']
        prediction = model.predict(features)

        return jsonify({'prediction': list(prediction)})
         
    except Exception as e:
        app.logger.error(f'Error: {e}')
        return jsonify({"error": "An error occurred during processing"}), 500

if __name__ == '__main__':
    
#  coordinates()    
 app.run(debug=True)
