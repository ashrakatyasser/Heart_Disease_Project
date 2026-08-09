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

# Build path relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
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
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

def build_input_form():
    st.sidebar.header("Input Patient Data")
    
    # Using proper datatypes so Pipeline handles imputers correctly
    age = st.sidebar.number_input("Age", min_value=20, max_value=100, value=50, step=1)
    sex = st.sidebar.selectbox("Sex (1=Male, 0=Female)", [0, 1])
    cp = st.sidebar.selectbox("Chest Pain Type (1–4)", [1, 2, 3, 4])
    trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120, step=1)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)
    restecg = st.sidebar.selectbox("Resting ECG (0,1,2)", [0, 1, 2])
    thalach = st.sidebar.number_input("Max Heart Rate Achieved", min_value=70, max_value=210, value=150, step=1)
    exang = st.sidebar.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
    oldpeak = st.sidebar.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
    slope = st.sidebar.selectbox("Slope (1,2,3)", [1, 2, 3])
    ca = st.sidebar.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia (3=Normal, 6=Fixed Defect, 7=Reversible Defect)", [3, 6, 7])

    # Convert to DataFrame with explicit float type to prevent SimpleImputer numpy dtype errors
    data_dict = {
        "age": float(age),
        "sex": float(sex),
        "cp": float(cp),
        "trestbps": float(trestbps),
        "chol": float(chol),
        "restecg": float(restecg),
        "thalach": float(thalach),
        "exang": float(exang),
        "oldpeak": float(oldpeak),
        "slope": float(slope),
        "ca": float(ca),
        "thal": float(thal)
    }

    return pd.DataFrame([data_dict], columns=SELECTED_FEATURES)

def main():
    st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
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

            if pred == 1:
                st.error(f"⚠️ Predicted Result: **High Risk of Heart Disease** (Class {int(pred)})")
            else:
                st.success(f"✅ Predicted Result: **Low Risk / Healthy** (Class {int(pred)})")

            st.write(f"Probability (Disease Present): **{pred_proba * 100:.1f}%**")

            # Chart representation
            chart_data = pd.DataFrame(
                {"Probability": [1 - pred_proba, pred_proba]},
                index=["Healthy", "Heart Disease Risk"]
            )
            st.bar_chart(chart_data)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("💡 Tip: Try retraining the model file using the exact same scikit-learn version as your app runtime.")

    st.markdown("---")
    st.subheader("Model metadata")
    st.write(f"Model expects {len(SELECTED_FEATURES)} features: `{SELECTED_FEATURES}`")

if __name__ == "__main__":
    main()
