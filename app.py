import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Load Ridge model
with open('ridge_model.pkl', 'rb') as f:
    model: Pipeline = pickle.load(f)

# Load important features, default values, and all feature names
with open('top_features.pkl', 'rb') as f:
    top_features = pickle.load(f)

with open('default_values.pkl', 'rb') as f:
    default_values = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    all_features = pickle.load(f)

st.title("🏠 House Price Predictor")

# Create input form
user_inputs = {}
st.subheader("Enter values for important features")

for feature in top_features:
    default = round(default_values.get(feature, 0), 2)
    user_inputs[feature] = st.number_input(f"{feature}", value=default)

# Fill in the remaining features with default values
input_data = default_values.copy()
input_data.update(user_inputs)

# Ensure feature order matches model input
input_df = pd.DataFrame([input_data])[all_features]

# Predict
if st.button("Predict Sale Price"):
    log_pred = model.predict(input_df)[0]
    sale_price = np.expm1(log_pred)  # since target is log-transformed
    st.success(f"🏡 Predicted Sale Price: ${sale_price:,.0f}")
