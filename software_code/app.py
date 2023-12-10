from flask import Flask, render_template, request, jsonify
import folium
from folium import ClickForMarker
import pandas as pd
import joblib
app = Flask(__name__)

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


# Load the model and scaler
clf_multi_param = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the dataset
df = pd.read_csv('data_20221_cleaned.csv')

from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

# Assuming df is your DataFrame
df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')

# Drop rows where either LATITUDE or LONGITUDE is NaN
df.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)

# Now fit the NearestNeighbors model
coordinates = df[['LATITUDE', 'LONGITUDE']].values
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(coordinates)


@app.route('/predict', methods=['POST'])

# @app.route('/predict', methods=['POST'])

# @app.route('/coordinates', methods=['POST'])
# var latitude = e.latlng.lat;
            # var longitude
def predict():
    data = request.json
    lat = data['latitude']
    lng = data['longitude']
   
    # Query the dataset based on latitude and longitude
    query_result = df[(df['LATITUDE'] == lat) & (df['LONGITUDE'] == lng)]

    # If exact coordinates are not found, find the nearest coordinates
    if query_result.empty:
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

    # return jsonify({"message": f"Water quality is {prediction_label}"}), 200
    
    return jsonify({
        "message": prediction_label,
        
    })

if __name__ == '__main__':
    # coordinates()
    app.run(debug=True)
