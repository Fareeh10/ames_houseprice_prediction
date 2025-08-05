import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib

if 'show_inputs' not in st.session_state:
    st.session_state.show_inputs = False

# Load model
model: Pipeline = joblib.load("ridge_model.pkl")

# Load features and values
with open('top_features.pkl', 'rb') as f:
    top_features = pickle.load(f)

with open('default_values.pkl', 'rb') as f:
    default_values = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    all_features = pickle.load(f)

# Binary one-hot encoded features
binary_features = [
    'CentralAir', 'BldgType_1Fam'
]

# Features that were log-transformed during training
log_transformed_features = [
    'MasVnrArea', 'BsmtFinSF1', '1stFlrSF',
    'GrLivArea', 'OpenPorchSF', 'EnclosedPorch', 'WoodDeckSF'
]

# Categorical features mapping (string -> numeric)
categorical_mappings = {
    'SaleCondition': {'Normal': 0, 'Partial': 1, 'Abnorml': 2, 'Family': 3, 'Alloca': 4, 'AdjLand': 5},
    'Functional': {'Typ': 0, 'Min2': 1, 'Min1': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6},
    'FireplaceQu': {'Excellent': 4, 'Good': 3, 'Typical/Average': 2, 'Fair': 1, 'Poor': 0},
    'KitchenQual': {'Excellent': 4, 'Good': 3, 'Typical/Average': 2, 'Fair': 1},
    'ExterQual': {'Excellent': 4, 'Good': 3, 'Typical/Average': 2, 'Fair': 1},
    'HeatingQC': {'Excellent': 4, 'Good': 3, 'Typical/Average': 2, 'Fair': 1, 'Poor': 0},
    'PavedDrive': {'Y': 2, 'P': 1, 'N': 0},
    "BsmtExposure": {'No':0, 'Average':1, 'Good':2,'Mb': 3},
    'BsmtQual': {'Excellent': 4, 'Good': 3, 'Typical/Average': 2, 'Fair': 1},
}


# Categorical / Ordinal encoded features with limited allowed values
categorical_features = {
    "OverallQual": list(range(1, 11)),
    "YearRemodAdd": list(range(1950, 2011)),
    "SaleCondition": [0, 1, 2, 3, 4],
    "Functional": [0, 1, 2],
    "HalfBath": [0, 1]
}

basement_features = ['BsmtQual', 'BsmtFinSF1', 'BsmtUnfSF', 'BsmtExposure', 'TotalBsmtSF']

# Set page configuration
st.set_page_config(page_title="Ames House Price Predictor", page_icon="🏠", layout="centered")

# Initialize session state
if "show_inputs" not in st.session_state:
    st.session_state.show_inputs = False

# Landing Page
# Landing Page
if not st.session_state.show_inputs:
    st.markdown(
        """
        <div style="background-color:#fff3cd; padding:10px; border-left:6px solid #ffa500; margin-bottom:20px;">
            <strong>🚧 Under Construction:</strong> This app is still being developed. UI/Some features may not be final.
        </div>

        <div style='text-align: center; padding: 80px 0;'>
            <h1 style='font-size: 3em;color:#FF4B4B'>Ames House Price Predictor</h1>
            <p style='font-size: 1.2em; color: gray;'>Predict house prices in Ames, Iowa using a trained ML model</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ✅ Streamlit button instead of raw HTML
    if st.button("Get Started", use_container_width=True):
        st.session_state.show_inputs = True
        st.rerun()

else:

    # Add a back button
    if st.button("Back to Home"):
        st.session_state.show_inputs = False
        st.rerun()

    # Section Header
    st.markdown("<h2 style='text-align: center;color:#FF4B4B'>Enter Property Details</h2>", unsafe_allow_html=True)
    st.markdown("")

    user_inputs = {}

    # Handle categorical string-to-numeric inputs
    for feature, options in categorical_mappings.items():
        reverse_map = {v: k for k, v in options.items()}
        default_raw = default_values.get(feature, 0)
        default_str = reverse_map.get(default_raw, list(options.keys())[0])
        user_input = st.selectbox(f"{feature}", list(options.keys()), index=list(options.keys()).index(default_str))
        user_inputs[feature] = options[user_input]

    # --- Other Features ---
    for feature in top_features:
        if feature in basement_features:
            continue  # already handled seprately

        if feature in binary_features and feature not in ['Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab','NridgHt','Somerst']:
            choice = st.radio(f"{feature}?", ["No", "Yes"], index=0, horizontal=True)
            user_inputs[feature] = 1 if choice == "Yes" else 0

        elif feature in categorical_features:
            options = categorical_features[feature]
            default_val = default_values.get(feature, options[0])
            index = options.index(default_val) if default_val in options else 0
            user_inputs[feature] = st.slider(f"{feature}", min_value=min(options), max_value=max(options), value=options[index])

        elif feature in log_transformed_features:
            log_val = default_values.get(feature, 0)
            normal_val_default = round(np.expm1(log_val))
            normal_val = st.number_input(f"{feature} (normal value)", value=normal_val_default, min_value=0)
            user_inputs[feature] = np.log1p(normal_val)

        else:
            default = round(default_values.get(feature, 0))
            user_inputs[feature] = st.number_input(f"{feature}", value=default)

    # --- Basement Features ---
    with st.expander("Basement Features"):
        
        for feature in basement_features:
            if feature in categorical_features:
                options = categorical_features[feature]
                default_val = default_values.get(feature, options[0])
                index = options.index(default_val) if default_val in options else 0
                user_inputs[feature] = st.slider(f"{feature}", min_value=min(options), max_value=max(options), value=options[index])
            elif feature in log_transformed_features:
                log_val = default_values.get(feature, 0)
                normal_val_default = round(np.expm1(log_val))
                normal_val = st.number_input(f"{feature} (normal value)", value=normal_val_default, min_value=0)
                user_inputs[feature] = np.log1p(normal_val)
            else:
                default = round(default_values.get(feature, 0))
                user_inputs[feature] = st.number_input(f"{feature}", value=default)

    foundation_options = ['CBlock', 'PConc', 'Slab', 'Other']
    foundation_feature_map = {
        'CBlock': 'Foundation_CBlock',
        'PConc': 'Foundation_PConc',
        'Slab': 'Foundation_Slab'
    }
    default_foundation = 'CBlock'

    selected_foundation = st.selectbox("Select Foundation Type", foundation_options, index=foundation_options.index(default_foundation))

    # Initialize all features to 0
    for feature in foundation_feature_map.values():
        user_inputs[feature] = 0

    # Set selected feature to 1 if it's not 'Other'
    if selected_foundation in foundation_feature_map:
        user_inputs[foundation_feature_map[selected_foundation]] = 1

    neighborhood_options = ['NridgHt', 'Somerst', 'Other']
    neighborhood_selection = st.selectbox("Select Neighborhood", neighborhood_options)

    user_inputs['NridgHt'] = 1 if neighborhood_selection == 'NridgHt' else 0
    user_inputs['Somerst'] = 1 if neighborhood_selection == 'Somerst' else 0

    # Fill in other features not shown to user
    input_data = default_values.copy()
    input_data.update(user_inputs)

    # Arrange inputs in correct order
    input_df = pd.DataFrame([input_data])[all_features]

    # Predict and display
    if st.button("🔍 Predict Sale Price"):
        log_price = model.predict(input_df)[0]
        final_price = np.expm1(log_price)
        st.success(f"🏡 Estimated Sale Price: **${final_price:,.0f}**")
