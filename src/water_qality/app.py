from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from flask_cors import CORS  # Import CORS
import logging
from sklearn.preprocessing import LabelEncoder

# app = Flask(__name__)
# CORS(app)  # Enable CORS on your Flask app
# from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # CORS applied to /coordinates route for all origins
logging.basicConfig(level=logging.INFO)
# MODEL5
path="E:\\GWL\\src\\water_qality\\finalized_model.pkl"
model5=joblib.load(path)
df5 = pd.read_csv('water_qality\\adjusted_water_quality_data_v2.csv')
df5['Latitude'] = pd.to_numeric(df5['Latitude'], errors='coerce')
df5['Longitude'] = pd.to_numeric(df5['Longitude'], errors='coerce')
df5.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Fit the NearestNeighbors model
coordinates = df5[['Latitude', 'Longitude']].values
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(coordinates)
# MODEL1
df=pd.read_csv('E:\\GWL\\src\\water_qality\\updated_dataset_with_water_well_suitability.csv')
model0=joblib.load('E:\\GWL\\src\\water_qality\\water_well_suitability_rf_model.joblib')
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

coordinates1 = df[['Latitude', 'Longitude']].values
neigh2 = NearestNeighbors(n_neighbors=1)
neigh2.fit(coordinates1)
# MODEL1



# MODEL2
path="E:\\datASET\\Usefulldata\\nw\\depth.pkl"
model2=joblib.load(path)
df2=pd.read_csv('E:\\datASET\\Usefulldata\\nw\\updated_gdf_well_soil_lithology_rainfall_with_lithology.csv')
# df.head()
df2['Latitude__well'] = pd.to_numeric(df2['Latitude__well'], errors='coerce')
df2['Longitude__well'] = pd.to_numeric(df2['Longitude__well'], errors='coerce')
df2.dropna(subset=['Latitude__well', 'Longitude__well'], inplace=True)
# df.head()
coordinates2 = df2[['Latitude__well', 'Longitude__well']].values
neigh5 = NearestNeighbors(n_neighbors=1)
neigh5.fit(coordinates2)
# MODEL3
model_path="C:\\Users\\admin\\Downloads\\rf_model_discharge.joblib"
encoders_path='C:\\Users\\admin\\Downloads\\label_encoders.joblib'


# MODEL_4
# df=pd.read_csv('E:\\datASET\\Usefulldata\\nw\\updated_dataset_with_water_well_suitability.csv')
model4=joblib.load('E:\\GWL\\src\\water_qality\\water_well_drilling_technique_predictor.joblib')

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
        query_result = df5.iloc[[nearest_index]]
        distances_, indices_ = neigh2.kneighbors(np.array([[lat, lng]]))
        nearest_index = indices_[0][0]
        result=df.iloc[[nearest_index]]
        # model2
        distances, indices = neigh5.kneighbors(np.array([[lat, lng]]))
        nearest_index = indices[0][0]
        result_2=df2.iloc[[nearest_index]]
        # MODEL1
        features = ['AQUIFER_TYPE', 'Broad_Soil_Type', 'WLS_WTR_LEVEL_Categorized', 'SITE_TYPE']
        X = result[features]
        label_encoders = {}
        for feature in features:
              label_encoders[feature] = LabelEncoder()
              X[feature] = label_encoders[feature].fit_transform(X[feature])
        suit_predictions = model0.predict(X)
        #       MODEL_2
        # distances, indices = neigh5.kneighbors(np.array([[lat, lng]]))
        # nearest_index = indices[0][0]
        # result_2=df2.iloc[[nearest_index]]
     
        columns_to_drop = ['Unnamed: 0', 'SITE_ID', 'STATE_NAME', 'DISTRICT_NAME', 'TAHSIL_NAME', 'BLOCK_NAME', 'Longitude__rainfall', 'Latitude__rainfall']
        relevant_df = result_2.drop(columns=columns_to_drop)
        result_2= result_2.drop('WLS_WTR_LEVEL', axis=1)
# y = df['WLS_WTR_LEVEL']
# result_2
        depth_predictions=model2.predict(result_2)
# # Separating the target variable and features
#         result_2 = result_2.drop('WLS_WTR_LEVEL', axis=1)
# # y = df['WLS_WTR_LEVEL']
#         depth_predictions=model2.predict(result_2)

        #  q   uery_result
       
        # MODEL3
        model3, label_encoders = load_model_and_encoders(model_path, encoders_path)
        # Preprocess the data
        data_=result[['AQUIFER_TYPE', 'SITE_TYPE', 'Broad_Soil_Type', 'Broader_Classification_Lithology','Total Annual Ground Water Recharge']]
        preprocessed_data = preprocess_data(data_, label_encoders)
        print(preprocess_data)

# Make a prediction
        prediction_discharge = make_prediction(model3, preprocessed_data)
        
        # MODEL4
        # X4=result[['WLS_WTR_LEVEL','AQUIFER_TYPE']]
        # X4['WLS_WTR_LEVEL_Categorized'] = X['WLS_WTR_LEVEL'].apply(categorize_water_level)
        # predicted_techniques = model.predict(X4)
      
        features1=query_result[['pH', 'Cl', 'NO3', 'TH',  'Ca', 'Mg','TDS']]
# ['pH', 'Cl', 'NO3', 'TH',  'Ca', 'Mg','TDS']
        prediction = model5.predict(features1)
        # return jsonify({'suit_predictions':suit_predictions})

        return jsonify({'prediction': list(prediction),'suit_predictions':list(suit_predictions),'depth_predictions':list(depth_predictions),
                        'prediction_discharge':list(prediction_discharge)})
         
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
def load_model_and_encoders(model_path, encoders_path):
    # Load the saved model
    model = joblib.load(model_path)

    # Load the saved label encoders
    label_encoders = joblib.load(encoders_path)

    return model, label_encoders

def preprocess_data(input_data, label_encoders):
    # Assuming input_data is a pandas DataFrame with the columns:
    # ['AQUIFER_TYPE', 'Total Annual Ground Water Recharge', 'SITE_TYPE', 'Broad_Soil_Type', 'Broader_Classification_Lithology']

    # Encode categorical variables using the saved label encoders
    for column in ['AQUIFER_TYPE', 'SITE_TYPE', 'Broad_Soil_Type', 'Broader_Classification_Lithology']:
        le = label_encoders[column]
        input_data[column] = le.transform(input_data[column])

    return input_data

def make_prediction(model, preprocessed_data):
    # Make a prediction
    prediction = model.predict(preprocessed_data)
    return prediction


if __name__ == '__main__':
    
#  coordinates()    
 app.run(debug=True)
