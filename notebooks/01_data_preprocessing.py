from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA_PATH = "data/heart_disease.csv"  # optional save
CLEANED_PATH = "data/heart_disease_cleaned.csv"
PREPROCESSOR_PATH = "models/preprocessor.joblib"

def load_data():
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets


    # If y is a DataFrame with its own column name(s), just rename the first one
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:  # single column
            y.columns = ["target"]
        else:
            # If multiple columns, keep them distinct
            y = y.rename(columns={col: f"target_{col}" for col in y.columns})
    else:  # if it's a Series
        y = y.rename("target")

    df = pd.concat([X, y], axis=1)
    return df


def build_preprocessor(df, scaler_type="standard"):
    # Keep only the features selected in feature selection
    selected_features = ['age', 'sex', 'cp', 'trestbps', 'chol',
                        'restecg', 'thalach', 'exang', 'oldpeak',
                        'slope', 'ca', 'thal']  # 12 features

    numeric_cols = [col for col in selected_features if col in df.columns]
    categorical_cols = []  # dataset is all numeric in this case


    num_imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
    num_pipeline = Pipeline([
        ("imputer", num_imputer),
        ("scaler", scaler),
    ])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_pipeline = Pipeline([
        ("imputer", cat_imputer),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols),
    ], remainder="drop")

    return preprocessor, numeric_cols, categorical_cols

def preprocess_and_save(df, scaler_type="standard"):
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(df, scaler_type=scaler_type)

    X = df.drop(columns=["target"]) if "target" in df.columns else df.copy()
    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)

    num_names = numeric_cols
    cat_names = []
    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        for i, col in enumerate(categorical_cols):
            cats = ohe.categories_[i]
            cat_names += [f"{col}__{c}" for c in cats]
    feature_names = num_names + cat_names

    df_clean = pd.DataFrame(X_transformed, columns=feature_names, index=df.index)
    if "target" in df.columns:
        df_clean["target"] = df["target"].values

    df_clean.to_csv(CLEANED_PATH, index=False)
    joblib.dump({
        "preprocessor": preprocessor,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_names": feature_names,
    }, PREPROCESSOR_PATH)

    print(f"Saved cleaned dataset to {CLEANED_PATH} and preprocessor to {PREPROCESSOR_PATH}")
    return df_clean

if __name__ == "__main__":
    df = load_data()
    df_clean = preprocess_and_save(df, scaler_type="standard")

