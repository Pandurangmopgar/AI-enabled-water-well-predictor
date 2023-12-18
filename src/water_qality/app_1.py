from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from flask_cors import CORS  # Import CORS
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
# import pandas as pd
# import numpy as np
# Importing necessary libraries for machine learning
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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
df5= pd.read_csv('water_qality\\adjusted_water_quality_data_v2.csv')

# Convert LATITUDE and LONGITUDE to numeric and drop NaN values
df5['Latitude'] = pd.to_numeric(df5['Latitude'], errors='coerce')
df5['Longitude'] = pd.to_numeric(df5['Longitude'], errors='coerce')
df5.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Fit the NearestNeighbors model
coordinates = df5[['Latitude', 'Longitude']].values
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(coordinates)
df=pd.read_csv('E:\\datASET\\Usefulldata\\nw\\updated_dataset_with_water_well_suitability.csv')
model=joblib.load('E:\\GWL\\src\\water_qality\\water_well_suitability_rf_model.joblib')
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

coordinates1 = df[['Latitude', 'Longitude']].values
neigh2 = NearestNeighbors(n_neighbors=1)
neigh2.fit(coordinates1)
# Model2



path="C:\\datASET\\Final\\updated_gdf_well_soil_lithology_rainfall_with_lithology.csv"
df=pd.read_csv("C:\\datASET\\Final\\updated_gdf_well_soil_lithology_rainfall_with_lithology.csv")

df['Latitude__well'] = pd.to_numeric(df['Latitude__well'], errors='coerce')
df['Longitude__well'] = pd.to_numeric(df['Longitude__well'], errors='coerce')
df.dropna(subset=['Latitude__well', 'Longitude__well'], inplace=True)
coordinates2 = df[['Latitude__well', 'Longitude__well']].values
neigh3= NearestNeighbors(n_neighbors=1)
neigh3.fit(coordinates2)
# MODEL_4
# df=pd.read_csv('E:\\datASET\\Usefulldata\\nw\\updated_dataset_with_water_well_suitability.csv')
model4=joblib.load('E:\\GWL\\src\\water_qality\\water_well_drilling_technique_predictor.joblib')

# Convert LATITUDE and LONGITUDE to numeric and drop NaN values
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
        distances_, indices_ = neigh2.kneighbors(np.array([[lat, lng]]))
        nearest_index = indices_[0][0]
        result=df.iloc[[nearest_index]]
        distances, indices__ = neigh2.kneighbors(np.array([[lat, lng]]))
        nearest_index = indices__[0][0]
        result_=df.iloc[[nearest_index]]
        features = ['AQUIFER_TYPE', 'Broad_Soil_Type', 'WLS_WTR_LEVEL_Categorized', 'SITE_TYPE']
        X = result[features]
        label_encoders = {}
        for feature in features:
              label_encoders[feature] = LabelEncoder()
              X[feature] = label_encoders[feature].fit_transform(X[feature])
#         #  q   uery_result
        suit_predictions = model.predict(X)
        #  q   uery_result
        suit_predictions = model.predict(X)
        # MODEL_2
        # columns_to_drop = ['Unnamed: 0', 'SITE_ID', 'STATE_NAME', 'DISTRICT_NAME', 'TAHSIL_NAME', 'BLOCK_NAME', 'Longitude__rainfall', 'Latitude__rainfall']
        # relevant_df = result_.drop(columns=columns_to_drop)
        # X = relevant_df.drop('WLS_WTR_LEVEL', axis=1)
        # y = relevant_df['WLS_WTR_LEVEL']
        # depth_prediction=model.predict(X)

# Separating the target variable and features
        # X = df.drop('WLS_WTR_LEVEL', axis=1)
        # # y = df['WLS_WTR_LEVEL']
        # predictions=model.predict(X)

        # MODEL4
        # X4=result[['WLS_WTR_LEVEL','AQUIFER_TYPE','Broad_Soil_Type']]
        # X4['WLS_WTR_LEVEL_Categorized'] = X['WLS_WTR_LEVEL'].apply(categorize_water_level)
        # predicted_techniques = model.predict(X4)
        # label_encoder = LabelEncoder()
        # X['AQUIFER_TYPE_encoded'] = label_encoder.fit_transform(X['AQUIFER_TYPE'])
        # X['Broad_Soil_Type_encoded'] = label_encoder.fit_transform(X['Broad_Soil_Type'])

        # features1 = ['WLS_WTR_LEVEL', 'AQUIFER_TYPE_encoded', 'Broad_Soil_Type_encoded']
        

        features=query_result[['pH', 'Cl', 'NO3', 'TH',  'Ca', 'Mg','TDS']]
# ['pH', 'Cl', 'NO3', 'TH',  'Ca', 'Mg','TDS']
        prediction = model.predict(features)
        return jsonify({'prediction': list(prediction),'suit_predictions':list(suit_predictions),'depth_prediction':'full_depth'})
         
        return jsonify({'prediction': list(prediction),'suit_predictions':list(suit_predictions)})
         
    except Exception as e:
        app.logger.error(f'Error: {e}')
        return jsonify({"error": "An error occurred during processing"}), 500

def categorize_water_level(water_level):
    if water_level > 50:  # Example threshold for 'Deep'
        return "Deep"
    elif water_level < 10:  # Example threshold for 'Shallow'
        return "Shallow"
    else:
        return "Artesian"

if __name__ == '__main__':
    
#  coordinates()    
 app.run(debug=True)