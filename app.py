import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("ridge_model.pkl")  # Replace with your actual .pkl filename

# Page title
st.markdown("""
<div style="background-color:#fff3cd; padding:10px; border-left:6px solid #ffa500; margin-bottom:20px;">
    <strong>🚧 Under Construction:</strong> This app is still being developed. Some features may not be final.
</div>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Ames House Price Predictor", layout="centered")
st.title("🏠 Ames House Price Predictor")
st.markdown("Enter the details below to predict the **house price in Ames, Iowa**.")

# Collect input features
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)
    gr_liv_area = st.number_input("Above Ground Living Area (sqft)", 300, 6000, 1500)
    garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)

with col2:
    total_bsmt_sf = st.number_input("Total Basement Area (sqft)", 0, 3000, 800)
    year_built = st.slider("Year Built", 1870, 2024, 2000)
    full_bath = st.slider("Full Bathrooms", 0, 4, 2)

# Prediction
if st.button("🔮 Predict House Price"):
    input_data = np.array([[overall_qual, gr_liv_area, garage_cars, total_bsmt_sf, year_built, full_bath]])
    predicted_price = model.predict(input_data)[0]
    st.success(f"🏡 Estimated Price: **${predicted_price:,.0f}**")
