"""
Streamlit UI (app.py)
- Load final_model.pkl pipeline
- Provide input widgets for features (12 selected ones)
- Show prediction and probability, plus sample visualizations
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Build path relative to this file (ui/app.py → ../models/final_model.pkl)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")

# Explicitly define the selected 12 features
SELECTED_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        return None

def build_input_form():
    st.sidebar.header("Input patient data")
    input_data = {}

    # Numeric inputs
    input_data["age"] = st.sidebar.number_input("Age", min_value=20, max_value=100, value=50)
    input_data["sex"] = st.sidebar.selectbox("Sex (1=Male, 0=Female)", [0, 1])
    input_data["cp"] = st.sidebar.selectbox("Chest Pain Type (1–4)", [1, 2, 3, 4])
    input_data["trestbps"] = st.sidebar.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    input_data["chol"] = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    input_data["restecg"] = st.sidebar.selectbox("Resting ECG (0,1,2)", [0, 1, 2])
    input_data["thalach"] = st.sidebar.number_input("Max Heart Rate Achieved", min_value=70, max_value=210, value=150)
    input_data["exang"] = st.sidebar.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
    input_data["oldpeak"] = st.sidebar.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
    input_data["slope"] = st.sidebar.selectbox("Slope (1,2,3)", [1, 2, 3])
    input_data["ca"] = st.sidebar.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    input_data["thal"] = st.sidebar.selectbox("Thalassemia (3=Normal, 6=Fixed Defect, 7=Reversible Defect)", [3, 6, 7])

    return pd.DataFrame([input_data], columns=SELECTED_FEATURES)

def main():
    st.title("❤️ Heart Disease Prediction")
    st.write("Enter patient features in the sidebar and press **Predict** to see results.")

    pipeline = load_pipeline()
    if pipeline is None:
        return

    input_df = build_input_form()

    if st.button("Predict"):
        try:
            pred_proba = pipeline.predict_proba(input_df)[:, 1][0]
            pred = pipeline.predict(input_df)[0]

            st.success(f"Predicted class: **{int(pred)}**")
            st.write(f"Probability (disease present): **{pred_proba:.3f}**")

            # simple visualization
            st.bar_chart({
                "No Disease": [1 - pred_proba],
                "Disease": [pred_proba]
            })

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.subheader("Model metadata")
    st.write(f"Model expects {len(SELECTED_FEATURES)} features: {SELECTED_FEATURES}")

if __name__ == "__main__":
    main()
