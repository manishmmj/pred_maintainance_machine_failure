import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Load the pre-trained model and scaler
MODEL_PATH = "best_model_logistic_regression.pkl"  # Update with the saved model path
SCALER_PATH = "scaler.pkl"  # Save and load the scaler used during training

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

# Visualization: ROC Curve
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(model.classes_, prediction_proba)  # Mocked classes for demonstration
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
st.pyplot(plt)

# Visualization: Confusion Matrix
st.subheader("Confusion Matrix")
y_true = [1, 0]  # Replace with actual test labels
y_pred = prediction  # Replace with predicted labels
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Failure", "Failure"])

fig, ax = plt.subplots()
disp.plot(ax=ax, cmap="Blues")
st.pyplot(fig)

st.write("Note: Replace `y_true` and `y_pred` with actual test data for real metrics.")


