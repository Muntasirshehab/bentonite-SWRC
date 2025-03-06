import streamlit as st
import joblib  # For loading the .pkl model
import numpy as np
from catboost import CatBoostRegressor  # Import CatBoostRegressor

# Streamlit UI
st.title("CatBoost Model Loader and Predictor")

# Load the model from the GitHub repository directory
model_path = "best_model.pkl"  # Make sure this matches your file name
try:
    model = joblib.load(model_path)

    # Check if the loaded model is a CatBoostRegressor
    if not isinstance(model, CatBoostRegressor):
        st.error("Loaded model is not a CatBoostRegressor! Please check your .pkl file.")
    else:
        st.success("Model successfully loaded!")

        # Input fields for making predictions
        st.subheader("Make a Prediction")

        # Feature names (as per your model)
        feature_names = [
            "Confined = 1 / Unconfined = 0",
            "Wetting = 1 / Drying = 0",
            "Specific Gravity (G_s)",
            "Dry Density (ρ_d)",
            "Montmorillonite Content (Mt.c)",
            "Initial Water Content (w_i)",
            "Initial Void Ratio (e_0)",
            "Plasticity Index (I_P)",
            "Suction (ψ)"
        ]

        input_features = []

        # Create number input fields with proper labels
        for feature in feature_names:
            val = st.number_input(f"{feature}", value=0.0)
            input_features.append(val)

        # Convert to numpy array for prediction
        input_array = np.array([input_features]).reshape(1, -1)

        if st.button("Predict"):
            # Make the prediction using CatBoostRegressor
            prediction = model.predict(input_array)

            # Display the prediction
            st.subheader("Prediction:")
            st.write(f"**Water content:** {prediction[0]}")

except FileNotFoundError:
    st.error("Model file not found! Make sure 'best_model.pkl' exists in the app directory.")
