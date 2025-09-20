"""
05_unsupervised_learning.py
- KMeans with elbow method
- Hierarchical clustering (dendrogram)
- Compare clusters with true labels if available
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

DATA_PATH = "data/heart_disease_reduced.csv"  # use reduced or cleaned

def run_clustering(max_k=10):
    df = pd.read_csv(DATA_PATH)
    y = df['target'].values if 'target' in df.columns else None
    X = df.drop(columns=['target']) if 'target' in df.columns else df
    # elbow
    inertias = []
    K_range = range(1, max_k+1)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.figure()
    plt.plot(K_range, inertias, marker='o')
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for K")
    plt.savefig("results/elbow_kmeans.png")
    plt.close()

    # choose k (example: 2 or 3)
    k_best = 2
    km = KMeans(n_clusters=k_best, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    if y is not None:
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)
        print("KMeans ARI:", ari, "NMI:", nmi)

    # hierarchical dendrogram
    z = linkage(X.sample(min(200, len(X))), method='ward')  # sample for faster plotting
    plt.figure(figsize=(10,6))
    dendrogram(z, truncate_mode='level', p=5)
    plt.title("Hierarchical Clustering Dendrogram (sampled)")
    plt.savefig("results/dendrogram.png")
    plt.close()

if __name__ == "__main__":
    run_clustering(max_k=8)
