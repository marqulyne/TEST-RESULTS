import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and encoders
model = joblib.load('investigation_test_results_model_with_metrics.joblib')
label_encoders = joblib.load('label_encoders_with_metrics.joblib')

# Function to extend the label encoder to handle unseen labels
def extend_label_encoder(encoder, new_value):
    new_classes = list(encoder.classes_)
    if new_value not in new_classes:
        new_classes.append(new_value)
        encoder.classes_ = np.array(new_classes)
    return encoder.transform([new_value])[0]

# Streamlit tabs
tab1, tab2 = st.tabs(["Prediction", "About the App"])

# Prediction tab
with tab1:
    st.title('Investigation Test Results Prediction')

    # Input fields for the prediction
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=120, value=25)
    location = st.text_input('Location/Ward/Village', 'Manyatta B')
    diagnoses = st.text_input('Diagnoses', 'Malaria')
    investigation_title = st.text_input('Investigation Title', 'Malaria blood smear')

    # Create new data for prediction
    new_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Location/Ward/Village': [location],
        'Diagnoses': [diagnoses],
        'Investigation titles': [investigation_title]
    })

    # Apply label encoders to categorical columns
    for col in ['Gender', 'Location/Ward/Village', 'Diagnoses', 'Investigation titles']:
        new_data[col] = extend_label_encoder(label_encoders[col], new_data[col][0])

    # Prediction button
    if st.button('Predict'):
        prediction = model.predict(new_data)
        
        # Map prediction result to either Positive or Negative
        result_map = {0: 'Negative', 1: 'Positive'}  # Ensure your model output matches this mapping
        predicted_result = result_map.get(prediction[0], "Unknown")  # Handle unexpected predictions
        st.write(f'Predicted Investigation Test Result: {predicted_result}')

# About tab
with tab2:
    st.title("About the App")

    st.subheader("Developers:")
    st.write("""
    - **Marquline Opiyo**
    - **Annrita Mukami**
    """)

    st.subheader("App Information:")
    st.write("""
    This application is designed to predict the results of investigation tests based on patient data.
    It uses machine learning models to analyze factors such as gender, age, location, diagnoses, and investigation test data to provide predictions.

    The prediction model was trained using real healthcare data and is aimed at assisting medical practitioners with quick insights into patient test results.
    """)

    st.subheader("Technology Stack:")
    st.write("""
    - **Python**: For model development and data processing.
    - **Streamlit**: For the interactive web application.
    - **scikit-learn**: For machine learning.
    """)
