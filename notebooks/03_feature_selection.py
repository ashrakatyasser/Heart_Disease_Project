"""
03_feature_selection.py
- Load cleaned dataset
- Compute RandomForest/XGBoost feature importances
- Apply RFE (with estimator)
- Run chi-square for categorical vs target (if target categorical/binary)
- Save a reduced dataset and importance plot
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.model_selection import train_test_split

CLEANED_PATH = "data/heart_disease_cleaned.csv"
REDUCED_PATH = "data/heart_disease_reduced.csv"
IMPORTANCE_PNG = "results/feature_importances.png"

def run_feature_selection(k_features=10):
    df = pd.read_csv(CLEANED_PATH)
    if 'target' not in df.columns:
        raise ValueError("target column not found in cleaned CSV")
    X = df.drop(columns=['target'])
    y = df['target']
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    feat_imp.to_csv("results/feature_importances.csv", index=False)

    plt.figure(figsize=(10,6))
    topn = feat_imp.head(30)
    plt.barh(topn['feature'][::-1], topn['importance'][::-1])
    plt.title("Top feature importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PNG)
    plt.close()

    # RFE with RF
    selector = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=k_features, step=0.1)
    selector.fit(X, y)
    selected_mask = selector.support_
    selected_features = X.columns[selected_mask].tolist()

    # Chi-square (requires non-negative ints) -> use SelectKBest on absolute if data is numeric scaled
    try:
        skb = SelectKBest(score_func=chi2, k=min(k_features, X.shape[1]))
        skb.fit(np.abs(X), y)
        chi_selected = X.columns[skb.get_support()].tolist()
    except Exception as e:
        chi_selected = []
        print("Chi2 selection skipped (requires non-negative features).", e)

    reduced_df = df[selected_features + ['target']]
    reduced_df.to_csv(REDUCED_PATH, index=False)
    joblib.dump({
        'selected_features': selected_features,
        'rfe_selector': selector
    }, "models/feature_selection.joblib")
    print(f"Saved reduced dataset to {REDUCED_PATH}")
    print("Top features:", selected_features[:20])
    return selected_features

if __name__ == "__main__":
    run_feature_selection(k_features=12)
