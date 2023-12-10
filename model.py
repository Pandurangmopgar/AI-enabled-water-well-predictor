# model = joblib.load('your_model.pkl')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Re-reading the dataset and skipping problematic lines
df = pd.read_csv('data_20221_cleaned.csv')

# Dropping columns with excessive missing values and those that are not relevant for water quality prediction
cols_to_drop = ['Well ID', 'S.No', 'STATE', 'DISTRICT', 'BLOCK', 'LOCATION', 'LATITUDE', 'LONGITUDE', 'Year', 'PO4', 'SiO2', 'TDS', 'U(ppb)']
df_clean = df.drop(columns=cols_to_drop)

# Convert all columns to numeric, coercing errors to NaN (Not a Number)
df_clean = df_clean.apply(pd.to_numeric, errors='coerce')

# Fill missing values with the median of each column
df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)

# Create a water quality label based on multiple parameters: pH, EC, TH, NO3, Na, K, F
# The labels are: "Good", "Moderate", "Poor"
conditions = [
    (df_clean['pH'] >= 6.5) & (df_clean['pH'] <= 8.5) &
    (df_clean['EC'] <= 1500) &
    (df_clean['TH'] <= 120) &
    (df_clean['NO3'] <= 10) &
    (df_clean['Na'] <= 200) &
    (df_clean['K'] <= 10) &
    (df_clean['F'] <= 1.5),
    (df_clean['pH'] < 6.5) | (df_clean['pH'] > 8.5) |
    (df_clean['EC'] > 1500) |
    (df_clean['TH'] > 120) |
    (df_clean['NO3'] > 10) |
    (df_clean['Na'] > 200) |
    (df_clean['K'] > 10) |
    (df_clean['F'] > 1.5)
]
choices = ['Good', 'Poor']
df_clean['Water_Quality'] = np.select(conditions, choices, default='Moderate')

# Convert the Water_Quality labels into numerical values
label_encoder = LabelEncoder()
df_clean['Water_Quality'] = label_encoder.fit_transform(df_clean['Water_Quality'])

# Split the data into features and target label
X = df_clean.drop('Water_Quality', axis=1)
y = df_clean['Water_Quality']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
clf_multi_param = RandomForestClassifier(random_state=42)
clf_multi_param.fit(X_train, y_train)

# Evaluate the model
y_pred = clf_multi_param.predict(X_test)
accuracy_multi_param = accuracy_score(y_test, y_pred)
classification_rep_multi_param = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
import joblib

# Save the model and scaler
joblib.dump(clf_multi_param, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
