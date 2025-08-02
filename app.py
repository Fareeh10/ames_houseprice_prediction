import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and feature list
model = joblib.load("ridge_model.pkl")
feature_names = joblib.load("model_features.pkl")
default_values = joblib.load("default_values.pkl")

# Page title
st.title("🏠 Ames House Price Prediction")

# User input for top important features
st.header("🔢 Enter Key House Details")
overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", value=1500)
garage_cars = st.slider("Garage Capacity (Cars)", 0, 4, 2)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", value=800)
year_built = st.number_input("Year Built", min_value=1870, max_value=2023, value=2000)
full_bath = st.slider("Full Bathrooms", 0, 4, 2)
half_bath = st.slider("Half Bathrooms", 0, 2, 1)
lot_area = st.number_input("Lot Area (sq ft)", value=9000)
tot_rms_abv_grd = st.slider("Total Rooms Above Ground", 2, 15, 6)
garage_area = st.number_input("Garage Area (sq ft)", value=500)

# Create input DataFrame using default values
input_data = pd.DataFrame([default_values])
input_data = input_data.reindex(columns=feature_names, fill_value=0)

# Overwrite with user inputs
input_data["OverallQual"] = overall_qual
input_data["GrLivArea"] = gr_liv_area
input_data["GarageCars"] = garage_cars
input_data["TotalBsmtSF"] = total_bsmt_sf
input_data["YearBuilt"] = year_built
input_data["FullBath"] = full_bath
input_data["HalfBath"] = half_bath
input_data["LotArea"] = lot_area
input_data["TotRmsAbvGrd"] = tot_rms_abv_grd
input_data["GarageArea"] = garage_area

# Add one-hot categorical defaults if needed (example)
# input_data["KitchenQual_TA"] = 1

# Predict when user clicks button
if st.button("Predict Price 💰"):
    prediction = model.predict(input_data)[0]
    st.success(f"🏡 Estimated House Price: **${prediction:,.0f}**")
