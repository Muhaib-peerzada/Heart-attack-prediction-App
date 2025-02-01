import pandas as pd
import streamlit as st
import joblib
import numpy as np

# Load saved models
log_model = joblib.load('logistic_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')

# Load the CSV file
df = pd.read_csv('heart_attack_data.csv')

# Streamlit UI
st.title("Heart Attack Prediction Model")
st.markdown("<h5 style='text-align: center; color: grey;'>by Peerzada Mubashir</h5>", unsafe_allow_html=True)

# Transparent box for the title and "by Peerzada Mubashir" only (without patient details)
st.markdown("""
    <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; background-color: transparent;">
    <h3>Enter patient details below:</h3>
""", unsafe_allow_html=True)

# User Inputs (All 9 Features)
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.radio("Sex", ("Male", "Female"))
total_cholesterol = st.number_input("Total Cholesterol Level", min_value=100, max_value=400, value=200)
ldl = st.number_input("LDL (Low-Density Lipoprotein)", min_value=50, max_value=300, value=100)
hdl = st.number_input("HDL (High-Density Lipoprotein)", min_value=20, max_value=100, value=50)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=90, max_value=200, value=120)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=130, value=80)
smoking = st.radio("Smoking Status", ("Non-Smoker", "Smoker"))
diabetes = st.radio("Diabetes Status", ("No", "Yes"))

# Convert categorical inputs to numerical values
sex = 1 if sex == "Male" else 0
smoking = 1 if smoking == "Smoker" else 0
diabetes = 1 if diabetes == "Yes" else 0

# Model Selection
model_choice = st.radio("Choose a Model:", ("Logistic Regression", "Random Forest"))

# Simplified note about the models
st.markdown("""
    **Model Comparison:**

    - **Logistic Regression:** 
      - Accuracy: 49% (better at catching heart attacks).
    
    - **Random Forest:** 
      - Accuracy: 64.5% (better at overall accuracy).

    **Recommendation:** Choose Logistic Regression if minimizing missed heart attacks is critical.
""", unsafe_allow_html=True)

# Define dataset description
dataset_description = """
This dataset contains 1,000 synthetic patient records generated for health risk assessment and predictive
modeling. It includes demographic, lifestyle, and biometric health indicators commonly used in cardiovascular
and general health research. Each record captures age, 
cholesterol levels, blood pressure, smoking habits, diabetes status, and heart attack history—key factors influencing cardiovascular diseases.

The dataset is used to develop predictive models to assess the risk of heart attacks based on these factors.
"""

# Create columns for layout (left side for dataset details)
col1, col2 = st.columns(2)

# Display dataset description in the left column
with col1:
    st.markdown("""
    <div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; background-color: transparent; width: 100%;">
    <h4>Dataset Details</h4>
    <p>{}</p>
    </div>
    """.format(dataset_description), unsafe_allow_html=True)

# Footer Note
st.markdown("""
    <div style="border-top: 2px solid #4CAF50; padding-top: 10px; text-align: center; font-size: 14px; color: grey;">
    <p><strong>Note:</strong> All the data used is synthetically generated and does not represent real patients. It is meant for research and educational purposes.</p>
    </div>
""", unsafe_allow_html=True)
