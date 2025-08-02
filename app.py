import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("ridge_model.pkl")

# Page config
st.set_page_config(page_title="Ames House Price Predictor", layout="centered")

st.markdown("""
<div style="background-color:#fff3cd; padding:10px; border-left:6px solid #ffa500; margin-bottom:20px;">
    <strong>🚧 Under Construction:</strong> This app is still being developed. Some features may not be final.
</div>
""", unsafe_allow_html=True)

st.title("🏠 Ames House Price Predictor")
st.markdown("Enter the details below to predict the **house price in Ames, Iowa**.")

# --- General Info ---
st.header("🏡 General Information")
col1, col2 = st.columns(2)
with col1:
    overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)
    year_built = st.slider("Year Built", 1870, 2024, 2000)
with col2:
    full_bath = st.slider("Full Bathrooms", 0, 4, 2)
    half_bath = st.slider("Half Bathrooms", 0, 2, 1)

# --- Size & Area ---
st.header("📏 Size Details")
col3, col4 = st.columns(2)
with col3:
    gr_liv_area = st.number_input("Above Ground Living Area (sqft)", 300, 6000, 1500)
    first_flr_sf = st.number_input("1st Floor SF", 300, 3000, 1200)
with col4:
    second_flr_sf = st.number_input("2nd Floor SF", 0, 3000, 400)
    total_bsmt_sf = st.number_input("Basement Area (sqft)", 0, 3000, 800)

# Derived bath counts
st.header("🛁 Bathrooms")
bsmt_full_bath = st.slider("Basement Full Baths", 0, 3, 1)
total_bath = full_bath + 0.5 * half_bath + bsmt_full_bath

# --- Prediction ---
if st.button("🔮 Predict House Price"):

    # Compute polynomial/interaction features manually (top ones)
    features = {
        "BsmtFullBath TotalBath": bsmt_full_bath * total_bath,
        "2ndFlrSF GrLivArea": second_flr_sf * gr_liv_area,
        "1stFlrSF GrLivArea": first_flr_sf * gr_liv_area,
        "FullBath TotalBath": full_bath * total_bath,
        "1stFlrSF 2ndFlrSF": first_flr_sf * second_flr_sf,
        "GrLivArea^2": gr_liv_area ** 2,
        "TotalBath^2": total_bath ** 2,
        "BsmtFullBath FullBath": bsmt_full_bath * full_bath,
        "HalfBath TotalBath": half_bath * total_bath,
        "TotalBath": total_bath,
        "2ndFlrSF^2": second_flr_sf ** 2,
        "FullBath HalfBath": full_bath * half_bath,
        "GrLivArea TotalBath": gr_liv_area * total_bath
        # Add more if used in training
    }

    input_data = np.array([list(features.values())])
    predicted_price = model.predict(input_data)[0]

    st.success(f"🏡 Estimated Price: **${predicted_price:,.0f}**")
