import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("ridge_model.pkl")  # Use your actual model

# Set page config and warning banner
st.set_page_config(page_title="Ames House Price Predictor", layout="centered")

st.markdown("""
<div style="background-color:#fff3cd; padding:10px; border-left:6px solid #ffa500; margin-bottom:20px;">
    <strong>🚧 Under Construction:</strong> This app is still being developed. Some features may not be final.
</div>
""", unsafe_allow_html=True)

st.title("🏠 Ames House Price Predictor")
st.markdown("Enter the details below to predict the **house price in Ames, Iowa**.")

# --- Group 1: General Information ---
st.header("🏡 General Information")
col1, col2 = st.columns(2)
with col1:
    overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)
    year_built = st.slider("Year Built", 1870, 2024, 2000)
with col2:
    full_bath = st.slider("Full Bathrooms", 0, 4, 2)
    half_bath = st.slider("Half Bathrooms", 0, 2, 1)

# --- Group 2: Area & Size ---
st.header("📏 Size Details")
col3, col4 = st.columns(2)
with col3:
    gr_liv_area = st.number_input("Above Ground Living Area (sqft)", 300, 6000, 1500)
    total_bsmt_sf = st.number_input("Total Basement Area (sqft)", 0, 3000, 800)
with col4:
    lot_area = st.number_input("Lot Area (sqft)", 1000, 50000, 8000)
    garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)

# --- Optional: Advanced Features in Expander ---
with st.expander("🔧 Advanced Options (Optional)"):
    col5, col6 = st.columns(2)
    with col5:
        kitchen_qual = st.selectbox("Kitchen Quality", ['Poor', 'Fair', 'Typical', 'Good', 'Excellent'])
        fireplace_qu = st.selectbox("Fireplace Quality", ['None', 'Poor', 'Fair', 'Typical', 'Good'])
    with col6:
        tot_rms_abv_grd = st.slider("Total Rooms Above Ground", 2, 15, 6)
        garage_area = st.number_input("Garage Area (sqft)", 0, 1500, 400)

    # Add more features below as needed

# --- Prediction Button ---
if st.button("🔮 Predict House Price"):
    # You’ll need to match the order and number of features exactly with training
    input_data = np.array([[
        overall_qual, gr_liv_area, garage_cars,
        total_bsmt_sf, year_built, full_bath,
        half_bath, lot_area, tot_rms_abv_grd, garage_area
        # Add more advanced features here in same order as model was trained
    ]])

    predicted_price = model.predict(input_data)[0]
    st.success(f"🏡 Estimated Price: **${predicted_price:,.0f}**")
