import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib

# Load model
model: Pipeline = joblib.load("ridge_model.pkl")

# Load feature data
with open('top_features.pkl', 'rb') as f:
    top_features = pickle.load(f)

with open('default_values.pkl', 'rb') as f:
    default_values = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    all_features = pickle.load(f)

st.title("🏠 House Price Predictor")

# Group top features manually (example)
feature_groups = {
    "🏗️ Overall Property Quality": ["OverallQual", "OverallCond"],
    "📏 Area and Size": ["GrLivArea", "TotalBsmtSF", "GarageArea", "LotArea"],
    "🏠 Rooms and Interior": ["TotRmsAbvGrd", "FullBath", "HalfBath", "BedroomAbvGr"],
    "🚗 Garage Details": ["GarageCars", "GarageYrBlt"],
    "📅 Year Built/Remodeled": ["YearBuilt", "YearRemodAdd"],
    "🌳 Neighborhood/Location": ["Neighborhood"]
}

user_inputs = {}
st.subheader("Enter property details")

# Input fields grouped
for group_name, features in feature_groups.items():
    with st.expander(group_name):
        for feature in features:
            if feature in top_features:
                default = round(default_values.get(feature, 0), 2)
                user_inputs[feature] = st.number_input(f"{feature}", value=default)

# Fill in the remaining top features not in groups (optional)
ungrouped_features = set(top_features) - set(f for flist in feature_groups.values() for f in flist)
if ungrouped_features:
    with st.expander("🧩 Other Important Features"):
        for feature in ungrouped_features:
            default = round(default_values.get(feature, 0), 2)
            user_inputs[feature] = st.number_input(f"{feature}", value=default)

# Fill in the remaining features with default values
input_data = default_values.copy()
input_data.update(user_inputs)

# Prepare input for prediction
input_df = pd.DataFrame([input_data])[all_features]

# Predict
if st.button("Predict Sale Price"):
    log_pred = model.predict(input_df)[0]
    sale_price = np.expm1(log_pred)  # convert log price to actual
    st.success(f"🏡 Predicted Sale Price: ${sale_price:,.0f}")
