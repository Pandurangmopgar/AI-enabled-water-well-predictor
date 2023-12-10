from flask import Flask, render_template, request, jsonify
import folium
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and scaler
clf_multi_param = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the dataset
df = pd.read_csv('data_20221_cleaned.csv')

@app.route('/')
def index():
    start_coords = (0, 0)
    folium_map = folium.Map(location=start_coords, zoom_start=4)
    return render_template('index.html', folium_map=folium_map._repr_html_())
@app.route('/coordinates', methods=['POST'])
def coordinates():
    data = request.json
    lat = data['lat']
    lng = data['lng']

    # Query the dataset based on latitude and longitude
    query_result = df[(df['LATITUDE'] == lat) & (df['LONGITUDE'] == lng)]

    # If exact coordinates are not found, find the nearest coordinates
    if query_result.empty:
        distances, indices =neigh.kneighbors(np.array([[lat, lng]]))
        nearest_index = indices[0][0]
        query_result = df.iloc[[nearest_index]]

    # Prepare the features for prediction
    features = query_result.drop(columns=['Well ID', 'S.No', 'STATE', 'DISTRICT', 'BLOCK', 'LOCATION', 'LATITUDE', 'LONGITUDE', 'Year', 'PO4', 'SiO2', 'TDS', 'U(ppb)'])
    features_scaled = scaler.transform(features)

    # Make a prediction
    prediction = clf_multi_param.predict(features_scaled)
    
    # Convert numerical prediction back to label
    label_map = {0: 'Good', 1: 'Moderate', 2: 'Poor'}
    prediction_label = label_map[prediction[0]]

    return jsonify({"message": f"Water quality is {prediction_label}"}), 20
if __name__ == '__main__':
    app.run(debug=True)
