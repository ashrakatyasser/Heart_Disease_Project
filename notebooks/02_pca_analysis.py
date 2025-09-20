"""
02_pca_analysis.py
- Load cleaned dataset (data/heart_disease_cleaned.csv)
- Apply PCA
- Save PCA transformer and PCA-transformed dataset
- Plot explained variance and 2D scatter (first 2 PCs)
"""

import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

CLEANED_PATH = "data/heart_disease_cleaned.csv"
PCA_PATH = "models/pca.joblib"
PCA_DATA_PATH = "data/pca_transformed.csv"

def run_pca(n_components=None):
    df = pd.read_csv(CLEANED_PATH)
    if 'target' in df.columns:
        X = df.drop(columns=['target']).values
        y = df['target'].values
    else:
        X = df.values
        y = None
    # If n_components is None, fit full PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    # explained variance
    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    # save
    joblib.dump(pca, PCA_PATH)
    cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=cols)
    if y is not None:
        df_pca['target'] = y
    df_pca.to_csv(PCA_DATA_PATH, index=False)
    # plots
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(evr)+1), evr, marker='o')
    plt.title("Explained Variance Ratio per Principal Component")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.grid(True)
    plt.savefig("results/pca_explained_variance.png", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(cumvar)+1), cumvar, marker='o')
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance")
    plt.grid(True)
    plt.savefig("results/pca_cumulative_variance.png", bbox_inches='tight')
    plt.close()

    # 2D scatter
    if X_pca.shape[1] >= 2:
        plt.figure(figsize=(8,6))
        if y is not None:
            for label in np.unique(y):
                idx = y == label
                plt.scatter(X_pca[idx,0], X_pca[idx,1], label=f"target={label}", alpha=0.6)
            plt.legend()
        else:
            plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.6)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA: PC1 vs PC2")
        plt.savefig("results/pca_scatter_2d.png", bbox_inches='tight')
        plt.close()
    print(f"PCA saved to {PCA_PATH}, transformed data to {PCA_DATA_PATH}")

if __name__ == "__main__":
    run_pca()
