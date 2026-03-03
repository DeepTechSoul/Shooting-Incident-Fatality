

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load the CSV file
data = pd.read_csv('NYPD_Shooting_Incident_Data__Historic_.csv')  # Replace with your actual file

# Preprocessing (adjust based on your dataset)
features = ['X_COORD_CD', 'Y_COORD_CD', 'Latitude', 'Longitude']
target = 'STATISTICAL_MURDER_FLAG'

# Split the dataset into X (features) and y (target)
X = data[features]
y = data[target]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier 
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Define a function to make predictions
def predict_fatality(features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return 'Fatal' if prediction[0] == 1 else 'Non-Fatal'

# Streamlit app interface
st.title("Shooting Incident Fatality Prediction")

# Input fields for the features
x_coord = st.number_input('X_COORD_CD', min_value=0, max_value=1000000, value=987500)
y_coord = st.number_input('Y_COORD_CD', min_value=0, max_value=1000000, value=202500)
latitude = st.number_input('Latitude', min_value=40.0, max_value=45.0, value=40.7128)
longitude = st.number_input('Longitude', min_value=-80.0, max_value=-70.0, value=-74.0060)

# Prepare input data for prediction
input_features = [x_coord, y_coord, latitude, longitude]

# Button to make prediction
if st.button('Predict Fatality'):
    result = predict_fatality(input_features)
    st.write(f"Prediction: {result}")

# Display model performance on the test set
if st.button('Evaluate Model'):
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("Model Performance on Test Data:")
    st.json(report)
