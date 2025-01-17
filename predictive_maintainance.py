import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model and scaler
MODEL_PATH = "best_model_lightgbm.pkl"  # Update with the saved model path

# load using pickle
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

scaler = pd.read_csv(train_data)

# Title and description
st.title("Predictive Maintenance Dashboard")
st.write("This app predicts machine failures based on input conditions and visualizes performance metrics.")

# Sidebar for user input
st.sidebar.header("Input Machine Parameters")

# Define user input fields
def user_input_features():
    air_temp = st.sidebar.slider("Air Temperature [K]", 290.0, 320.0, 300.0)
    process_temp = st.sidebar.slider("Process Temperature [K]", 295.0, 350.0, 310.0)
    rotational_speed = st.sidebar.slider("Rotational Speed [rpm]", 1000, 5000, 3000)
    torque = st.sidebar.slider("Torque [Nm]", 10, 100, 50)
    tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 300, 150)
    type_mapping = {"Low": 0, "Medium": 1, "High": 2}
    machine_type = st.sidebar.selectbox("Machine Type", options=["Low", "Medium", "High"])
    machine_type_encoded = type_mapping[machine_type]
    
    data = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type": machine_type_encoded
    }
    return pd.DataFrame([data])

# Get user inputs
input_df = user_input_features()

# Display input data
st.subheader("Machine Parameters")
st.write(input_df)

# Scale the input data

scaled_input = scaler.transform(input_df)

# Make predictions
if st.button("Predict"):
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)[:, 1]
    
    result = "Failure" if prediction[0] == 1 else "No Failure"
    st.subheader(f"Prediction: {result}")
    st.write(f"Failure Probability: {prediction_proba[0]:.2f}")

