import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model and scaler
MODEL_PATH = "best_model_lightgbm.pkl"  # Update with the saved model path
SCALER_PATH = "scaler.pkl"  # Save and load the scaler used during training


# Correct way to load using pickle
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


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

# Visualization: Basic ROC Curve
st.subheader("ROC Curve")
fpr = np.linspace(0, 1, num=100)  # Generate a dummy FPR
tpr = np.power(fpr, 2)  # Generate a dummy TPR for demonstration
roc_auc = np.trapz(tpr, fpr)  # Calculate the area under the curve

# Plot the ROC Curve using Streamlit
roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr})
st.line_chart(roc_data.rename(columns={"FPR": "False Positive Rate", "TPR": "True Positive Rate"}))

st.write(f"ROC AUC: {roc_auc:.2f}")

# Visualization: Confusion Matrix
st.subheader("Confusion Matrix")
y_true = [1, 0]  # Replace with actual test labels
y_pred = prediction.tolist()  # Replace with predicted labels
cm = np.array([
    [np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0)),  # True Negative
     np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))], # False Positive
    [np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0)),  # False Negative
     np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))]  # True Positive
])

st.write("Confusion Matrix:")
st.write(pd.DataFrame(cm, index=["Actual No Failure", "Actual Failure"], columns=["Predicted No Failure", "Predicted Failure"]))





