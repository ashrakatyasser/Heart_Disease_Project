"""
Combine preprocessing pipeline + best model in a single pipeline and save it (.pkl)
"""

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

PREPROCESSOR_PATH = "models/preprocessor.joblib"
BEST_MODEL_PATH = "models/rf_best.joblib"  # or whichever was best
FINAL_PIPELINE_PATH = "models/final_model.pkl"

def build_final_pipeline():
    preproc_bundle = joblib.load(PREPROCESSOR_PATH)
    preprocessor = preproc_bundle['preprocessor']
    model = joblib.load(BEST_MODEL_PATH)
    final_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    joblib.dump(final_pipe, FINAL_PIPELINE_PATH)
    print(f"Saved final pipeline to {FINAL_PIPELINE_PATH}")

if __name__ == "__main__":
    build_final_pipeline()
