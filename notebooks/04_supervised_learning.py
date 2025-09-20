"""
04_supervised_learning.py
- Train logistic regression, decision tree, random forest, SVM
- Evaluate: accuracy, precision, recall, f1, ROC & AUC
- Save models and a summary metrics CSV
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt

REDUCED_PATH = "data/heart_disease_reduced.csv"  # or cleaned if not run FS
MODEL_DIR = "models/"
METRICS_CSV = "results/evaluation_metrics.csv"

def load_dataset(path=None):
    path = path or REDUCED_PATH
    df = pd.read_csv(path)
    X = df.drop(columns=['target'])
    y = (df['target'] > 0).astype(int) 
    return X, y

def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    metrics = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        metrics.append({
            'model': name,
            'accuracy': acc,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'auc': auc
        })
        joblib.dump(model, f"{MODEL_DIR}{name}.joblib")
        # ROC plot
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure()
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
            plt.plot([0,1],[0,1],"--")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"ROC Curve - {name}")
            plt.legend()
            plt.savefig(f"results/roc_{name}.png")
            plt.close()
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(METRICS_CSV, index=False)
    return df_metrics

if __name__ == "__main__":
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    metrics = train_and_evaluate(X_train, X_test, y_train, y_test)
    print(metrics)
