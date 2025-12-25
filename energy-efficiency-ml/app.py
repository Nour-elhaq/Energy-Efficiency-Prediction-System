
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page config
st.set_page_config(
    page_title="Energy Efficiency Predictor",
    page_icon="⚡",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_artifacts():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'model_rf.pkl')
    scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    return model, scaler

model, scaler = load_artifacts()

st.title("⚡ Energy Efficiency Prediction")
st.markdown("""
This application predicts the **Heating Load** and **Cooling Load** of a building based on its parameters.
Data source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/242/energy+efficiency).
""")

if model is None:
    st.error("Model not found! Please run `src/train_model.py` to generate the model first.")
else:
    # Sidebar for inputs
    st.sidebar.header("Building Parameters")
    
    def user_input_features():
        X1 = st.sidebar.slider("Relative Compactness", 0.6, 1.0, 0.75)
        X2 = st.sidebar.slider("Surface Area", 500.0, 850.0, 600.0)
        X3 = st.sidebar.slider("Wall Area", 200.0, 450.0, 300.0)
        X4 = st.sidebar.slider("Roof Area", 100.0, 250.0, 220.0)
        X5 = st.sidebar.slider("Overall Height", 3.5, 10.0, 7.0)
        X6 = st.sidebar.selectbox("Orientation", [2, 3, 4, 5], format_func=lambda x: f"Orientation {x}")
        X7 = st.sidebar.slider("Glazing Area", 0.0, 0.4, 0.1)
        X8 = st.sidebar.selectbox("Glazing Area Distribution", [0, 1, 2, 3, 4, 5])
        
        data = {
            'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 
            'X5': X5, 'X6': X6, 'X7': X7, 'X8': X8
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    st.subheader("Building Parameters")
    st.write(input_df)

    # Predict
    if st.button("Predict"):
        # Scale inputs
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"Heating Load: {prediction[0][0]:.2f} kWh/m²")
        
        with col2:
            st.info(f"Cooling Load: {prediction[0][1]:.2f} kWh/m²")

    # Show Feature Importance (for RF)
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importance = model.feature_importances_
        feature_names = ['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area', 
                         'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution']
        # Since it's multi-output, RF might have estimators_ for each target or average. 
        # For simplicity in this visualization, we skip if it's complex standard RF return.
        # But wait, sklearn RF regressor for multi-output averages importances over outputs usually?
        # Let's check. If not, we can catch the error.
        try:
            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            imp_df = imp_df.sort_values(by='Importance', ascending=False)
            st.bar_chart(imp_df.set_index('Feature'))
        except:
            pass
