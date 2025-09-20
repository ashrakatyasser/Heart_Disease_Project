"""
06_hyperparameter_tuning.py
- Use GridSearchCV and RandomizedSearchCV for best params (RandomForest and SVM)
- Save best estimator
"""

import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import randint
import numpy as np

# Use built-in string scorers (works for multiclass and binary)
scoring = "roc_auc_ovr"

DATA_PATH = "data/heart_disease_reduced.csv"

def tune_random_forest(X, y):
    rf = RandomForestClassifier(random_state=42)
    param_dist = {
        'n_estimators': randint(50, 400),
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 6)
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rs = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=40,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )
    rs.fit(X, y)
    joblib.dump(rs.best_estimator_, "models/rf_best.joblib")
    print("Best RF params:", rs.best_params_)
    print("Best RF score:", rs.best_score_)
    return rs

def tune_svm(X, y):
    svc = SVC(probability=True, random_state=42)
    param_grid = {
        'C': [0.1, 1, 10, 50],
        'kernel': ['rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    gs = GridSearchCV(
        svc,
        param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )
    gs.fit(X, y)
    joblib.dump(gs.best_estimator_, "models/svm_best.joblib")
    print("Best SVM params:", gs.best_params_)
    print("Best SVM score:", gs.best_score_)
    return gs

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['target'])
    y = df['target']
    tune_random_forest(X, y)
    tune_svm(X, y)
