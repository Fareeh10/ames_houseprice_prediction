import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib

# Set page configuration first
st.set_page_config(page_title="Ames House Price Predictor", page_icon="🏠", layout="wide")

# --- CSS STYLING (WITH THE REQUESTED CHANGE) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .main .block-container {
        padding: 0rem;   /* Reduced from 2rem 3rem */
        background-color: #F0F2F6;
    }
    .landing-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 0 1rem;        /* Prevent horizontal overflow on small screens */
        margin: 0 auto;
        overflow: hidden;       /* Prevent scrollbars */
    }
    .headline { font-size: 3.5rem; line-height:4rem ;font-weight: 700; color: #2c3e50; margin-bottom: 1.5rem; }
    .subheadline { font-size: 1.3rem; color: #34495e; margin-bottom: 2rem; max-width: 600px; }
    .stButton>button {
        background-color: #cf1cff;
        color: white;
        font-size: 1.1em;
        font-weight: 600;
        padding: 0.8em 2.5em;
        border-radius: 50px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(189, 75, 255, 0.3);
    }
    .stButton>button:hover {
        background-color: #cf1cff;
        color: white; /* This line ensures the text remains white on hover */
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(189, 75, 255, 0.3);
    }
    h1 { text-align: center; color: #2c3e50; }
    h3 { color: #34495e; }
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
    }
    .prediction-card { text-align: center; padding: 30px; }
    .prediction-header { font-size: 1.5rem; font-weight: 600; color: #34495e; margin-bottom: 10px; }
    .prediction-value { font-size: 3rem; font-weight: 700; color: #cf1cff; }
            
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        border-left: 6px solid #ffa502;
        padding: 1rem 1.5rem;
        margin-bottom: 2rem;
        font-size: 1rem;
        font-weight: 500;
        border-radius: 8px;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .warning-box strong {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --- DATA LOADING ---
@st.cache_data
def load_data():
    model = joblib.load("ridge_model.pkl")
    with open('top_features.pkl', 'rb') as f:
        top_features = pickle.load(f)
    with open('default_values.pkl', 'rb') as f:
        default_values = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        all_features = pickle.load(f)
    return model, top_features, default_values, all_features

model, top_features, default_values, all_features = load_data()


# --- FEATURE DEFINITIONS ---
categorical_mappings = {
    'SaleCondition': {'Normal': 0, 'Partial': 1, 'Abnorml': 2, 'Family': 3, 'Alloca': 4, 'AdjLand': 5},
    'Functional': {'Typ': 0, 'Min2': 1, 'Min1': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6},
    'FireplaceQu': {'Excellent': 4, 'Good': 3, 'Typical/Average': 2, 'Fair': 1, 'Poor': 0, 'No Fireplace': -1},
    'KitchenQual': {'Excellent': 4, 'Good': 3, 'Typical/Average': 2, 'Fair': 1},
    'ExterQual': {'Excellent': 4, 'Good': 3, 'Typical/Average': 2, 'Fair': 1},
    'HeatingQC': {'Excellent': 4, 'Good': 3, 'Typical/Average': 2, 'Fair': 1, 'Poor': 0},
    'PavedDrive': {'Y': 2, 'P': 1, 'N': 0},
    "BsmtExposure": {'Good': 3, 'Average': 2, 'Mn': 1, 'No': 0, 'No Basement': -1},
    'BsmtQual': {'Excellent': 4, 'Good': 3, 'Typical/Average': 2, 'Fair': 1, 'No Basement': -1},
}

# --- HELPER FUNCTION ---
def get_default_value(feature_name, fallback, is_int=True):
    val = default_values.get(feature_name, fallback)
    if isinstance(val, (list, pd.Series, np.ndarray)):
        val = val[0]
    return int(val) if is_int else float(val)

# --- SESSION STATE ---
if 'show_inputs' not in st.session_state:
    st.session_state.show_inputs = False
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# --- UI RENDERING ---
if not st.session_state.show_inputs:
    st.markdown("""
        <div class="warning-box">
            🚧 <strong>Under Construction:</strong> This app is still being developed. UI/Some features may not be final.
        </div>
        <div class="landing-container">
            <div class="headline">Ames House Price Predictor</div>
            <div class="subheadline">
                Unlock accurate property valuations in Ames, Iowa.
                Enter a few details and get an instant price estimate.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        if st.button("Get Started", use_container_width=True):
            st.session_state.show_inputs = True
            st.rerun()

else:
    if st.button("Back to Home"):
        st.session_state.show_inputs = False
        st.session_state.prediction = None 
        st.rerun()

    st.markdown("<h1>Property Details</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #34495e;'>Enter the property information below to get a price prediction.</p>", unsafe_allow_html=True)

    user_inputs = {}
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        st.subheader("Key Features")
        col1, col2 = st.columns(2)
        with col1:
            user_inputs['OverallQual'] = st.slider("Overall Quality", 1, 10, get_default_value('OverallQual', 7), 1, help="Rates the overall material and finish of the house (1-10).")
        with col2:
            normal_val_default = round(np.expm1(get_default_value('GrLivArea', 7.2, is_int=False)))
            normal_val = st.number_input("Above Ground Living Area (sq. ft)", value=normal_val_default, min_value=0, step=50, help="Total square feet of living space above ground.")
            user_inputs['GrLivArea'] = np.log1p(normal_val)

        col1, col2 = st.columns(2)
        with col1:
            if 'YearBuilt' in all_features:
                user_inputs['YearBuilt'] = st.number_input("Year Built", 1800, 2025, get_default_value('YearBuilt', 2005), help="Original construction date.")
        with col2:
            user_inputs['YearRemodAdd'] = st.slider("Year Remodeled", 1950, 2011, get_default_value('YearRemodAdd', 2005), help="Remodel date (same as construction date if no remodeling).")

        tab1, tab2, tab3, tab4 = st.tabs(["Exterior & Location", "Area Details", "Quality & Condition", "Rooms & Utilities"])

        with tab1:
            st.markdown("<h3>Location & Construction</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                neighborhood_options = ['NridgHt', 'Somerst', 'StoneBr', 'Other']
                neighborhood_selection = st.selectbox("Neighborhood", neighborhood_options, help="The physical location of the property within Ames.")
                for neighborhood in ['NridgHt', 'Somerst', 'StoneBr']:
                     user_inputs[neighborhood] = 1 if neighborhood_selection == neighborhood else 0
            with col2:
                foundation_options = ['Poured Concrete', 'Cinder Block', 'Slab', 'Other']
                foundation_map = {'Poured Concrete': 'Foundation_PConc', 'Cinder Block': 'Foundation_CBlock', 'Slab': 'Foundation_Slab'}
                selected_foundation_str = st.selectbox("Foundation Type", foundation_options, help="Type of foundation.")
                for key, val in foundation_map.items():
                    user_inputs[val] = 1 if selected_foundation_str == key else 0
            
            col1, col2 = st.columns(2)
            with col1:
                options = list(categorical_mappings['PavedDrive'].keys())
                drive_str = st.selectbox("Paved Driveway", options, index=0)
                user_inputs['PavedDrive'] = categorical_mappings['PavedDrive'][drive_str]
            with col2:
                normal_val_default = round(np.expm1(get_default_value('MasVnrArea', 0, is_int=False)))
                normal_val = st.number_input("Masonry Veneer Area (sq. ft)", value=normal_val_default, min_value=0, step=10, help="Masonry veneer area in square feet. Enter 0 if none.")
                user_inputs['MasVnrArea'] = np.log1p(normal_val)

        with tab2:
            st.markdown("<h3>Basement Details</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                options = list(categorical_mappings['BsmtQual'].keys())
                bsmt_qual_str = st.selectbox("Basement Quality", options, index=0, help="Evaluates the height of the basement.")
                user_inputs['BsmtQual'] = categorical_mappings['BsmtQual'][bsmt_qual_str]
            with col2:
                options = list(categorical_mappings['BsmtExposure'].keys())
                bsmt_exp_str = st.selectbox("Basement Exposure", options, index=3, help="Refers to walkout or garden level walls.")
                user_inputs['BsmtExposure'] = categorical_mappings['BsmtExposure'][bsmt_exp_str]
            
            col1, col2 = st.columns(2)
            with col1:
                normal_val_default = round(np.expm1(get_default_value('BsmtFinSF1', 0, is_int=False)))
                normal_val = st.number_input("Finished Basement Area (sq. ft)", value=normal_val_default, min_value=0, step=50, help="Type 1 finished square feet of basement area.")
                user_inputs['BsmtFinSF1'] = np.log1p(normal_val)
            with col2:
                user_inputs['TotalBsmtSF'] = st.number_input("Total Basement Area (sq. ft)", value=get_default_value('TotalBsmtSF', 864), min_value=0, step=50, help="Total square feet of basement area.")
                
            st.markdown("<h3 style='margin-top: 1.5rem;'>Porch & Deck Area</h3>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                normal_val_default = round(np.expm1(get_default_value('WoodDeckSF', 0, is_int=False)))
                normal_val = st.number_input("Wood Deck (sq. ft)", value=normal_val_default, min_value=0, key='WoodDeckSF')
                user_inputs['WoodDeckSF'] = np.log1p(normal_val)
            with col2:
                normal_val_default = round(np.expm1(get_default_value('OpenPorchSF', 0, is_int=False)))
                normal_val = st.number_input("Open Porch (sq. ft)", value=normal_val_default, min_value=0, key='OpenPorchSF')
                user_inputs['OpenPorchSF'] = np.log1p(normal_val)
            with col3:
                normal_val_default = round(np.expm1(get_default_value('EnclosedPorch', 0, is_int=False)))
                normal_val = st.number_input("Enclosed Porch (sq. ft)", value=normal_val_default, min_value=0, key='EnclosedPorch')
                user_inputs['EnclosedPorch'] = np.log1p(normal_val)

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                options = list(categorical_mappings['ExterQual'].keys())
                ext_qual_str = st.selectbox("Exterior Quality", options, index=1, help="Evaluates the quality of the material on the exterior.")
                user_inputs['ExterQual'] = categorical_mappings['ExterQual'][ext_qual_str]
            with col2:
                options = list(categorical_mappings['KitchenQual'].keys())
                kit_qual_str = st.selectbox("Kitchen Quality", options, index=1, help="Quality of the kitchen.")
                user_inputs['KitchenQual'] = categorical_mappings['KitchenQual'][kit_qual_str]

            col1, col2 = st.columns(2)
            with col1:
                options = list(categorical_mappings['HeatingQC'].keys())
                heat_qc_str = st.selectbox("Heating Quality/Condition", options, index=0, help="Overall quality and condition of the heating system.")
                user_inputs['HeatingQC'] = categorical_mappings['HeatingQC'][heat_qc_str]
            with col2:
                options = list(categorical_mappings['FireplaceQu'].keys())
                fire_qu_str = st.selectbox("Fireplace Quality", options, index=len(options)-1, help="Quality of the fireplace. Select 'No Fireplace' if applicable.")
                user_inputs['FireplaceQu'] = categorical_mappings['FireplaceQu'][fire_qu_str]

            col1, col2 = st.columns(2)
            with col1:
                options = list(categorical_mappings['Functional'].keys())
                func_str = st.selectbox("Home Functionality", options, index=0, help="Home functionality rating (Assume typical unless deductions).")
                user_inputs['Functional'] = categorical_mappings['Functional'][func_str]
            with col2:
                options = list(categorical_mappings['SaleCondition'].keys())
                sc_str = st.selectbox("Sale Condition", options, index=0, help="Condition of sale.")
                user_inputs['SaleCondition'] = categorical_mappings['SaleCondition'][sc_str]
        
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                if 'FullBath' in all_features:
                     user_inputs['FullBath'] = st.number_input("Full Bathrooms", 0, 5, get_default_value('FullBath', 2), help="Number of full bathrooms above grade.")
            with col2:
                user_inputs['HalfBath'] = st.radio("Half Bathrooms", [0, 1, 2], index=get_default_value('HalfBath', 1), horizontal=True, help="Number of half bathrooms above grade.")

            col1, col2 = st.columns(2)
            with col1:
                if 'BedroomAbvGr' in all_features:
                    user_inputs['BedroomAbvGr'] = st.number_input("Bedrooms Above Grade", 0, 8, get_default_value('BedroomAbvGr', 3), help="Number of bedrooms not in the basement.")
            with col2:
                if 'TotRmsAbvGrd' in all_features:
                    user_inputs['TotRmsAbvGrd'] = st.number_input("Total Rooms Above Grade", 2, 15, get_default_value('TotRmsAbvGrd', 6), help="Total rooms above grade (does not include bathrooms).")

            st.markdown("<h3 style='margin-top: 1.5rem;'>Utilities</h3>", unsafe_allow_html=True)
            user_inputs['CentralAir'] = 1 if st.toggle('Central Air Conditioning', value=True) else 0

        st.markdown('</div>', unsafe_allow_html=True) 

    st.markdown("<br>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 1.5, 1])
    with col2:
        if st.button("🔍 Predict Sale Price", use_container_width=True):
            input_data = default_values.copy()
            input_data.update(user_inputs)
            input_df = pd.DataFrame([input_data])[all_features]

            try:
                log_price = model.predict(input_df)[0]
                final_price = np.expm1(log_price)
                st.session_state.prediction = final_price
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.session_state.prediction = None

    if st.session_state.prediction is not None:
        st.markdown(f"""
        <div class="card prediction-card">
            <div class="prediction-header">Estimated Sale Price</div>
            <div class="prediction-value">${st.session_state.prediction:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
