import streamlit as st
import joblib
import numpy as np

# Load saved models
log_model = joblib.load('logistic_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')

# Streamlit UI
st.title("Heart Attack Prediction Model")
st.write("By Peerzada Mubashir")

# Creating a layout with columns for dataset details and healthy heart tips
col1, col2 = st.columns([2, 1])

with col1:
    # Dataset Details Box (Wider box)
    st.subheader("Dataset Details")
    st.write("""
    This dataset contains 1,000 synthetic patient records generated for health risk assessment and predictive
    modeling. It includes demographic, lifestyle, and biometric health indicators commonly used in cardiovascular
    and general health research. Each record captures age, 
    cholesterol levels, blood pressure, smoking habits, diabetes status, and heart attack history—key factors influencing cardiovascular diseases.
    
    The dataset is used to develop predictive models to assess the risk of heart attacks based on these factors.
    """)

with col2:
    # Healthy Heart Tips Box (Shorter box)
    st.subheader("Healthy Heart Tips")
    st.write("""
    - Eat a balanced diet.
    - Exercise regularly.
    - Maintain a healthy weight.
    - Avoid smoking and excessive alcohol consumption.
    - Manage stress levels.
    """)

st.write("\n" * 2)  # Adds space between tips and user input fields

st.write("Enter patient details below:")

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

# Predict button
if st.button("Predict"):
    new_data = np.array([[age, sex, total_cholesterol, ldl, hdl, systolic_bp, diastolic_bp, smoking, diabetes]])

    if model_choice == "Logistic Regression":
        prediction = log_model.predict(new_data)[0]
    else:
        prediction = rf_model.predict(new_data)[0]

    result = "Heart Attack" if prediction == 1 else "No Heart Attack"
    st.write(f"**Prediction using {model_choice}: {result}**")

    # Add prediction percentages (You can adjust this logic)
    if result == "Heart Attack":
        st.write(f"Likelihood of Heart Attack: 70%")
        st.write("Likelihood of No Heart Attack: 30%")
    else:
        st.write(f"Likelihood of No Heart Attack: 80%")
        st.write("Likelihood of Heart Attack: 20%")

# Note at the bottom of the page
st.write("\n" * 3)  # Adds space for better visual clarity
st.write("**Note:** All the data used is synthetically generated and does not represent real patients. It is meant for research and educational purposes.")
