import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from src.meta_learner import MetaLearnerNN
from preprocess import load_encoders
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to load models
@st.cache(allow_output_mutation=True)
def load_models():
    rf_model = joblib.load("models/random_forest.pkl")
    xgb_model = joblib.load("models/xgboost.pkl")
    svr_model = joblib.load("models/svr.pkl")
    meta_learner = load_meta_learner("models/meta_learner.pth", input_dim=3)
    return rf_model, xgb_model, svr_model, meta_learner


# Function to load the meta-learner
def load_meta_learner(model_path, input_dim):
    model = MetaLearnerNN(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Function to encode new data
def encode_new_data(data, encoders):
    for column, encoder in encoders.items():
        if column in data.columns:
            try:
                data[column] = encoder.transform(data[column].astype(str))
            except ValueError:
                data[column] = data[column].fillna('Unknown')
                data.loc[~data[column].isin(encoder.classes_), column] = 'Unknown'
                data[column] = encoder.transform(data[column].astype(str))
    return data


# Load encoders
encoders = load_encoders('path/to/encoders.pkl')

# Streamlit app title
st.title("Property Value Prediction App")

# Required Fields Section (Highlighted)
st.header("Required Fields")
property_subtype_en = st.selectbox(
    "Property Subtype",
    options=[
        'Flat', 'Villa', 'Hotel Apartment', 'Hotel Rooms', 'Government Housing', 'Residential',
        'Shop', 'Office', 'Commercial', 'Industrial', 'Land', 'Residential Flats', 'Hotel',
        'Airport', 'Stacked Townhouses', 'Unit', 'Sports Club', 'School', 'Agricultural',
        'General Use', 'Labor Camp', 'Clinic', 'Building', 'Petrol Station',
        'Commercial / Offices / Residential', 'Show Rooms', 'Workshop', 'Electricity Station',
        'Warehouse', 'Hospital', 'Gymnasium', 'Sized Partition', 'Residential / Attached Villas',
        'Health Club', 'Residential / Villas', 'Exhbition Center', 'Consulate'
    ],
    index=0
)
rooms_en = st.selectbox(
    "Number of Rooms (Default: 1 B/R)",
    options=['1 B/R', '2 B/R', '3 B/R', 'Studio', '4 B/R', '5 B/R', '6 B/R', 'Office', 
             'PENTHOUSE', 'Shop', '7 B/R', 'Single Room', 'Hotel'],
    index=0
)
area_en = st.text_input("Area")
property_size_sqm = st.number_input("Property Size (sqm)", min_value=1.0, step=1.0, value=50.0)
property_type_en = st.selectbox("Property Type", options=['Unit', 'Building', 'Land'], index=0)

# Optional Fields Section (Defaults Provided)
st.header("Optional Fields")
is_freehold_text = st.selectbox("Is Freehold?", options=[True, False], index=0)
project_name_en = st.text_input("Project Name", value="Unknown")
property_usage_en = st.selectbox("Property Usage", options=['Residential', 'Commercial'], index=0)
total_buyer = st.number_input("Total Buyer", min_value=1, step=1, value=1)
transaction_type_en = st.selectbox("Transaction Type", options=['Sales', 'Mortgage', 'Gifts'], index=0)
total_seller = st.number_input("Total Seller", min_value=1, step=1, value=1)
registration_type_en = st.selectbox("Registration Type", options=['Ready', 'Off-Plan'], index=0)

# Derive `is_offplan` based on `registration_type_en`
is_offplan = True if registration_type_en == 'Off-Plan' else False

# Not Required Fields (Auto Handled or Ignored)
st.header("Not Required Fields")
st.write("The following fields are not required and handled automatically:")
st.write("- **is_offplan**: Automatically derived from Registration Type.")
st.write("- **nearest_landmark_en**: Predicted using KMeans.")
st.write("- **nearest_mall_en**: Predicted using KMeans.")
st.write("- **nearest_metro_en**: Predicted using KMeans.")

# Prepare input data
input_data = {
    "property_subtype_en": property_subtype_en,
    "rooms_en": rooms_en,
    "area_en": area_en,
    "property_size_sqm": property_size_sqm,
    "property_type_en": property_type_en,
    "is_freehold_text": is_freehold_text,
    "project_name_en": project_name_en,
    "property_usage_en": property_usage_en,
    "total_buyer": total_buyer,
    "transaction_type_en": transaction_type_en,
    "total_seller": total_seller,
    "registration_type_en": registration_type_en,
    "is_offplan": is_offplan
}
input_df = pd.DataFrame([input_data])

# Load models
rf_model, xgb_model, svr_model, meta_learner = load_models()

# Prediction logic
if st.button("Predict"):
    try:
        # Encode data
        input_df_encoded = encode_new_data(input_df, encoders)

        # Base model predictions
        rf_pred = rf_model.predict(input_df_encoded)
        xgb_pred = xgb_model.predict(input_df_encoded)
        svr_pred = svr_model.predict(input_df_encoded)

        # Meta-learner prediction
        base_predictions = torch.tensor([[rf_pred[0], xgb_pred[0], svr_pred[0]]], dtype=torch.float32)
        meta_pred = meta_learner(base_predictions).item()

        # Display predictions
        st.subheader("Predictions")
        st.write(f"Random Forest Prediction: {rf_pred[0]:,.2f}")
        st.write(f"XGBoost Prediction: {xgb_pred[0]:,.2f}")
        st.write(f"SVR Prediction: {svr_pred[0]:,.2f}")
        st.write(f"Meta-Learner Final Prediction: {meta_pred:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
