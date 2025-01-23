import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Load the pre-trained model and scaler
MODEL_PATH = "model_XgbClassifier.pkl"  # Update with the saved model path

# load using pickle
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

scaler = pd.read_csv("train_data.csv")

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
# These values should be calculated and saved during training
means = {
    "Air temperature [K]": 298.15,
    "Process temperature [K]": 307.15,
    "Rotational speed [rpm]": 3000,
    "Torque [Nm]": 50,
    "Tool wear [min]": 150,
    "Type": 1  # Encoded type mean
}

stds = {
    "Air temperature [K]": 5.0,
    "Process temperature [K]": 10.0,
    "Rotational speed [rpm]": 1000,
    "Torque [Nm]": 20.0,
    "Tool wear [min]": 75,
    "Type": 0.5
}

# Scale the input manually
scaled_input = (input_df - pd.Series(means)) / pd.Series(stds)
scaled_input_np = scaled_input.to_numpy()  # Convert to NumPy array

# Ensure proper shape for single input
scaled_input_np = scaled_input_np.reshape(1, -1) if scaled_input_np.ndim == 1 else scaled_input_np

# Debugging: Log the shape of the input
st.write("Scaled Input Shape:", scaled_input_np.shape)

# make prediction
if st.button("Predict"):
    try:
        # Ensure scaled_input is in the correct format
        scaled_input_np = scaled_input.to_numpy()
        scaled_input_np = scaled_input_np.reshape(1, -1) if scaled_input_np.ndim == 1 else scaled_input_np

        # Make predictions
        prediction = model.predict(scaled_input_np)
        prediction_proba = model.predict_proba(scaled_input_np)[:, 1]

        # Display results
        result = "Failure" if prediction[0] == 1 else "No Failure"
        st.subheader(f"Prediction: {result}")
        st.write(f"Failure Probability: {prediction_proba[0]:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
    
    

# Visualization: Basic ROC Curve
st.subheader("ROC Curve")
fpr = np.linspace(0, 1, num=100)  # Generate a dummy FPR
tpr = np.power(fpr, 2)  # Generate a dummy TPR for demonstration
roc_auc = np.trapz(tpr, fpr)  # Calculate the area under the curve

# Plot the ROC Curve using Streamlit
roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr})
st.line_chart(roc_data.rename(columns={"FPR": "False Positive Rate", "TPR": "True Positive Rate"}))

st.write(f"ROC AUC: {roc_auc:.2f}")

