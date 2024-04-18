from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from flask_cors import CORS  # Import CORS
import logging
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from statistics import mode
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from collections import Counter

# from flask import Flask, request, jsonify
# from flask_sqlalchemy import SQLAlchemy
# app = Flask(__name__)
# CORS(app)  # Enable CORS on your Flask app
# from flask_cors import CORS

app = Flask(__name__)
# CORS(app, resources={r"/predict": {"origins": "*"}}) 
# # CORS applied to /coordinates route for all origins
CORS(app, resources={r"/predict": {"origins": ["http://localhost:5000"]}})
logging.basicConfig(level=logging.INFO)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
# db = SQLAlchemy(app)

# MODEL5
from collections import Counter

# Function to convert degrees to radians
def deg2rad(deg):
    return deg * (np.pi/180)

# Load the dataset
file_path = 'C:\\Users\\admin\\Downloads\\Aquifer data_Cuddalore (1).xlsx'  # Replace with the actual path to your dataset
data = pd.read_excel(file_path)

# Separate the dataset into SR and HR subsets
sr_data = data[data['FORMATION'] == 'SR']
hr_data = data[data['FORMATION'] == 'HR']

# Prepare the data for BallTree with Haversine metric
sr_coordinates = sr_data[['Y_IN_DEC', 'X_IN_DEC']].apply(deg2rad).values
hr_coordinates = hr_data[['Y_IN_DEC', 'X_IN_DEC']].apply(deg2rad).values

# Create BallTrees for SR and HR
sr_tree = BallTree(sr_coordinates, metric='haversine')
hr_tree = BallTree(hr_coordinates, metric='haversine')

# Function to predict the aquifer range using the BallTree models
def predict_aquifer_range(latitude, longitude, rock_type):
    # Convert input coordinates to radians for the tree query
    query_point = np.radians(np.array([latitude, longitude]).reshape(1, -1))
    radius = 10 / 6371  # 10 km radius in radians

    if rock_type == 'SR':
        indices = sr_tree.query_radius(query_point, r=radius)
        relevant_data = sr_data
        columns = ['Aq_I_range', 'Aq_II_range', 'Aq_III_range', 'Aq_IV_range']
    elif rock_type == 'HR':
        indices = hr_tree.query_radius(query_point, r=radius)
        relevant_data = hr_data
        columns = ['Aq_I_range', 'Aq_II_range']
    else:
        return "Invalid rock type"

    # Predicting the aquifer range for each level
    predicted_ranges = {}
    if len(indices[0]) > 0:
        for column in columns:
            if column in relevant_data.columns:
                ranges = relevant_data[column].iloc[indices[0]]
                most_common_range = Counter(ranges).most_common(1)[0][0]
                predicted_ranges[column] = most_common_range
        return predicted_ranges
    else:
        return "No neighbors within 10 km"
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib

# "C:\Users\admin\Downloads\sr_tree_model.joblib"
# MODEL_4
#\\GWL\\src\\water_qality\\water_well_drilling_technique_predictor.joblib')
model1=joblib.load("E:\\datASET\\Usefulldata\\nw\\another\\predicted_techniques_hr .joblib")
file_path = "C:\\Users\\admin\\Downloads\\Aquifer data_Cuddalore (1).xlsx" 
df1=pd.read_excel(file_path)
sr_data = df1[df1['FORMATION'] == 'SR']
hr_data = df1[df1['FORMATION'] == 'HR']
# model5
model5=joblib.load("E:\\datASET\\Usefulldata\\nw\\another\\predicted_labels_sr_corrected.joblib")
model5_1=joblib.load("E:\\datASET\\Usefulldata\\nw\\another\\predicted_labels_hr_corrected.joblib")
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
        predicted_ranges = predict_aquifer_range(lat, lng, 'SR')
    
        #  q   uery_result
        data2 = pd.read_excel('C:\\Users\\admin\\Downloads\\Aquifer data_Cuddalore (1).xlsx')  # Replace with the path to your dataset

# Step 2: Fill missing values
        data2['aq1_yield (lps)'].fillna(2, inplace=True)
        data2['aq2_yield (lps)'].fillna(4, inplace=True)
        data2['AQ3_yield (lps)'].fillna(8, inplace=True)
        data2['AQ4_yield (lps)'].fillna(16, inplace=True)

# Step 3: Filter the dataset for Soft Rock (SR) and Hard Rock (HR)
        sr_data = data2[data2['FORMATION'] == 'SR']
        hr_data = data2[data2['FORMATION'] == 'HR']

# Step 4: Select relevant columns for SR and HR
        sr_discharge_columns = ['Y_IN_DEC', 'X_IN_DEC', 'aq1_yield (lps)', 'aq2_yield (lps)', 'AQ3_yield (lps)', 'AQ4_yield (lps)']
        hr_discharge_columns = ['Y_IN_DEC', 'X_IN_DEC', 'aq1_yield (lps)', 'aq2_yield (lps)']

# Step 5: Extract relevant data for SR and HR
        sr_discharge_data = sr_data[sr_discharge_columns]
        hr_discharge_data = hr_data[hr_discharge_columns]
        nn_sr_discharge = joblib.load("E:\\datASET\\Usefulldata\\nw\\another\\nn_model_sr_discharge.joblib")
        nn_hr_discharge = joblib.load("E:\\datASET\\Usefulldata\\nw\\another\\nn_model_sr_discharge.joblib")
        verage_discharge_sr = predict_discharge([lat,lng], nn_sr_discharge, sr_discharge_data)
     # average_discharge_hr = predict_discharge(example_coordinates_hr, nn_hr_discharge, hr_discharge_data)
       
     

# Make a prediction
        # prediction_discharge = make_prediction(model3, preprocessed_data)
        # result=result[['AQUIFER_TYPE','WLS_WTR_LEVEL','Broader_Classification_Lithology','Broader_Classification_Soil','Total Annual Ground Water Recharge']]
        
# Applying the mapping function to SR and HR datasets
        sr_data['Drilling_Techniques'] = sr_data.apply(lambda row: map_drilling_techniques(row, 'SR'), axis=1)
        hr_data['Drilling_Techniques'] = hr_data.apply(lambda row: map_drilling_techniques(row, 'HR'), axis=1)
        predicted_techniques_sr = knn_interpolate_drilling_techniques([lat,lng], sr_data, 'SR')
        # predicted_techniques_hr = knn_interpolate_drilling_techniques(, hr_data, 'HR')
        # Filtering the dataset for Soft Rock (SR) and Hard Rock (HR)
        sr_data_1= df1[df1['FORMATION'] == 'SR']
        hr_data_1 = df1[df1['FORMATION'] == 'HR']

# Selecting relevant columns for SR and HR water quality features
        sr_water_quality_columns = ['Y_IN_DEC', 'X_IN_DEC', 'EC (mS/cm)', 'F (mg/l)', 
                            'EC (mS/cm).1', 'F  (mg/l)', 'EC (mS/cm).2', 'F  (mg/l).1', 
                            'EC (mS/cm).3', 'F  (mg/l).2']
        hr_water_quality_columns = ['Y_IN_DEC', 'X_IN_DEC', 'EC (mS/cm).4', 'F  (mg/l).3', 
                            'EC (mS/cm).5', 'F  (mg/l).4']

# Extracting relevant data for SR and HR
        sr_water_quality_data = sr_data_1[sr_water_quality_columns]
        hr_water_quality_data = hr_data_1[hr_water_quality_columns]

# Imputing missing values using KNN
        knn_imputer = KNNImputer(n_neighbors=5)
        imputed_sr_water_quality = knn_imputer.fit_transform(sr_water_quality_data)
        imputed_hr_water_quality = knn_imputer.fit_transform(hr_water_quality_data)

# Converting imputed data back to DataFrame
        imputed_sr_df = pd.DataFrame(imputed_sr_water_quality, columns=sr_water_quality_columns)
        imputed_hr_df = pd.DataFrame(imputed_hr_water_quality, columns=hr_water_quality_columns)
        imputed_sr_df['Water_Quality_Labels'] = imputed_sr_df.apply(assign_water_quality_labels, axis=1)
        imputed_hr_df['Water_Quality_Labels'] = imputed_hr_df.apply(assign_water_quality_labels, axis=1)

# Training Nearest Neighbors models for SR and HR
        nn_model_sr = NearestNeighbors(n_neighbors=5)
        nn_model_hr = NearestNeighbors(n_neighbors=5)
        nn_model_sr.fit(imputed_sr_df[['Y_IN_DEC', 'X_IN_DEC']])
        nn_model_hr.fit(imputed_hr_df[['Y_IN_DEC', 'X_IN_DEC']])
        
        predicted_labels_sr = predict_water_quality([lat,lng], 'SR', nn_model_sr, imputed_sr_df)
        # predicted_labels_hr = predict_water_quality([lat,lng], 'HR', nn_model_hr, imputed_hr_df)



#         features1=query_result[['pH', 'Cl', 'NO3', 'TH',  'Ca', 'Mg','TDS']]
# # ['pH', 'Cl', 'NO3', 'TH',  'Ca', 'Mg','TDS']
#         prediction = model5.predict(features1)
        # return jsonify({'suit_predictions':suit_predictions})

        return jsonify({'suitability_predictions':'suitable', 'depth_predictions': predicted_ranges ,'prediction_discharge':list( verage_discharge_sr), 'drilling_techniques':list( predicted_techniques_sr),'water_quality': list(predicted_labels_sr)
       })
    except Exception as e:
        app.logger.error(f'Error: {e}')
        return jsonify({"error": "An error occurred during processing"}), 500


def predict_discharge(coordinates, nn_model, discharge_data):
    distances, indices = nn_model.radius_neighbors([coordinates])
    if len(indices[0]) > 0:
        nearest_discharges = discharge_data.iloc[indices[0], 2:]
        return nearest_discharges.mean()
    else:
        return "No nearby data within 10 km radius"


def assign_suitability_labels(depth, rock_type):
    if rock_type == 'SR':
        return 'Suitable' if depth < 150 else 'Not Suitable'
    elif rock_type == 'HR':
        return 'Suitable' if depth < 100 else 'Not Suitable'

def predict_suitability_knn(new_coordinates, features, labels, nn_model):
    distances, indices = nn_model.kneighbors([new_coordinates])
    nearest_labels = labels.iloc[indices[0]]
    prediction = mode(nearest_labels)
    return prediction














def assign_water_quality_labels(row):
    labels = []
    for i in range(0, len(row), 2):  # EC and F pairs
        ec, f = row[i], row[i+1]
        if ec < 1500 and f < 1.5:
            labels.append('Good')
        elif 1500 <= ec <= 3000 or 1.5 <= f <= 3.0:
            labels.append('Moderate')
        else:
            labels.append('Poor')
    return labels

    
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

def predict_water_quality(coordinates, formation, nn_model, water_quality_df):
    distances, indices = nn_model.kneighbors([coordinates])
    nearest_labels = water_quality_df.iloc[indices[0]]['Water_Quality_Labels']
    averaged_labels = [mode(labels) for labels in zip(*nearest_labels)]
    return averaged_labels


def assign_drilling_technique(rock_type, aquifer_level):
    if rock_type == 'SR':
        if aquifer_level in [1, 2]:
            return 'Rotary Drilling'
        elif aquifer_level in [3, 4]:
            return 'Rotary Drilling with Casing'
    elif rock_type == 'HR':
        if aquifer_level == 1:
            return 'Hammer Drilling'
        elif aquifer_level == 2:
            return 'Percussive Drilling'
    return 'Unknown'

# Function to map each row to its corresponding drilling techniques
def map_drilling_techniques(row, rock_type):
    techniques = []
    if rock_type == 'SR':
        for level in range(1, 5):  # SR has 4 aquifer levels
            techniques.append(assign_drilling_technique('SR', level))
    elif rock_type == 'HR':
        for level in range(1, 3):  # HR has 2 aquifer levels
            techniques.append(assign_drilling_technique('HR', level))
    return techniques

# Function to perform KNN interpolation for drilling techniques
def knn_interpolate_drilling_techniques(new_coordinates, data, rock_type, num_neighbors=5):
    # Setting up the Nearest Neighbors model
    nn = NearestNeighbors(n_neighbors=num_neighbors)
    nn.fit(data[['Y_IN_DEC', 'X_IN_DEC']])

    # Finding the nearest neighbors for the new coordinates
    distances, indices = nn.kneighbors([new_coordinates])
    nearest_points = data.iloc[indices[0]]

    # Determining the most common drilling technique among the nearest neighbors
    if rock_type == 'SR':
        num_levels = 4  # Number of aquifer levels in SR
    elif rock_type == 'HR':
        num_levels = 2  # Number of aquifer levels in HR

    aggregated_techniques = []
    for level in range(num_levels):
        techniques_at_level = nearest_points['Drilling_Techniques'].apply(lambda x: x[level])
        most_common_technique = mode(techniques_at_level)
        aggregated_techniques.append(most_common_technique)

    return aggregated_techniques

if __name__ == '__main__':
    app.run(debug=True)
