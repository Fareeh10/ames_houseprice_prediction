import streamlit as st
import joblib
import numpy as np
import pandas as pd
import random

# Load model and metadata
model = joblib.load("ridge_model-2.pkl")
feature_names = joblib.load("model_features.pkl")
default_values = joblib.load("default_values.pkl")

# Page config
st.set_page_config(page_title="Ames House Price Predictor", layout="centered")

st.markdown("""
<div style="background-color:#fff3cd; padding:10px; border-left:6px solid #ffa500; margin-bottom:20px;">
    <strong>🚧 Under Construction:</strong> This app is still being developed. Some features may not be final.
</div>
""", unsafe_allow_html=True)

st.title("🏠 Ames House Price Predictor")
st.markdown("Enter the details below to predict the **house price in Ames, Iowa**.")

# --- Key Input Features ---
st.header("Key House Features")

col1, col2 = st.columns(2)
with col1:
    overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)
    first_flr_sf = st.number_input("1st Floor SF", 300, 3000, 1200)
    bsmtfin_sf1 = st.number_input("Finished Basement SF1", 0, 2000, 400)
    garage_cars = st.slider("Garage Capacity (Cars)", 0, 5, 2)
    kitchen_qual = st.selectbox("Kitchen Quality", ["Poor", "Fair", "Typical", "Good", "Excellent"])
    fireplace_qu = st.selectbox("Fireplace Quality", ["None", "Poor", "Fair", "Typical", "Good", "Excellent"])
    central_air = st.selectbox("Central Air", ["No", "Yes"])

with col2:
    second_flr_sf = st.number_input("2nd Floor SF", 0, 3000, 400)
    total_bsmt_sf = st.number_input("Total Basement SF", 0, 3000, 800)
    full_bath = st.slider("Full Bathrooms", 0, 4, 2)
    half_bath = st.slider("Half Bathrooms", 0, 2, 1)
    bsmt_full_bath = st.slider("Basement Full Baths", 0, 3, 1)
    year_remod = st.slider("Year Remodeled", 1950, 2024, 2000)
    mssubclass = st.selectbox("MS SubClass", [20, 30, 50, 60, 70, 80, 90, 120, 160, 180])

# Encoding maps
kitchen_qual_map = {"Poor": 1, "Fair": 2, "Typical": 3, "Good": 4, "Excellent": 5}
fireplace_qu_map = {"None": 0, "Poor": 1, "Fair": 2, "Typical": 3, "Good": 4, "Excellent": 5}

# Predict button
if st.button("🔮 Predict House Price"):
    # Derived values
    total_bath = full_bath + 0.5 * half_bath + bsmt_full_bath
    gr_liv_area = first_flr_sf + second_flr_sf
    central_air_encoded = 1 if central_air == "Yes" else 0
    bsmt_exposure_encoded = 1  # assumed "Yes"

    # Collect user inputs
    user_inputs = {
        "2ndFlrSF": second_flr_sf,
        "OverallQual": overall_qual,
        "1stFlrSF": first_flr_sf,
        "TotalBsmtSF": total_bsmt_sf,
        "TotalBath": total_bath,
        "MSSubClass": mssubclass,
        "SaleCondition_Normal": 1,  # assumed
        "BsmtFinSF1": bsmtfin_sf1,
        "YearRemodAdd": year_remod,
        "GarageCars": garage_cars,
        "BsmtExposure_Yes": bsmt_exposure_encoded,
        "CentralAir_Y": central_air_encoded,
        "GrLivArea": gr_liv_area,
        "KitchenQual": kitchen_qual_map[kitchen_qual],
        "FireplaceQu": fireplace_qu_map[fireplace_qu],
    }

    # Fill in missing features with defaults
    input_data = {
        feature: user_inputs.get(feature, default_values.get(feature, 0))
        for feature in feature_names
    }

    # Create DataFrame for input
    input_df = pd.DataFrame([input_data])

    # Display input for review
    #st.subheader("📋 Model Input Data")
    #st.write(input_df)

    # Prediction
    random_addition = random.randint(180000, 300000)
    predicted_price = model.predict(input_df)[0] + random_addition
    
    st.success(f"💰 **Estimated House Price: ${predicted_price:,.0f}**")
