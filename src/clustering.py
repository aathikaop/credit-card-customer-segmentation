# Import Libraries

import pandas as pd
import joblib

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

import matplotlib.pyplot as plt


# Set after reviewing the elbow
N_CLUSTERS = 4


# find optimal k for KMeans

def find_optimal_k(
        X,
        k_range=range(2, 11)
):

    inertia = []
    silhouette = []

    for k in k_range:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init="auto"
        )

        labels = model.fit_predict(X)

        inertia.append(model.inertia_)
        silhouette.append(silhouette_score(X, labels))

    return list(k_range), inertia, silhouette


def plot_elbow_and_silhouette(
        k_values,
        inertia,
        silhouette,
        save_path=None
):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(k_values, inertia, marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(k_values, silhouette, marker="o", color="orange")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.close()


# KMeans

def perform_kmeans(
        X,
        n_clusters
):

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"
    )

    labels = model.fit_predict(X)

    return model, labels


# Hierarchical

def perform_hierarchical(
        X,
        n_clusters
):

    model = AgglomerativeClustering(
        n_clusters=n_clusters
    )

    labels = model.fit_predict(X)

    return model, labels


#  Gaussian Mixture Model

def perform_gmm(
        X,
        n_components
):

    model = GaussianMixture(
        n_components=n_components,
        random_state=42
    )

    labels = model.fit_predict(X)

    return model, labels


# DBSCAN

def perform_dbscan(
        X,
        eps=2.5,
        min_samples=10
):

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = model.fit_predict(X)

    return model, labels


# Run all models together

def run_clustering(
        X,
        n_clusters
):

    kmeans_model, kmeans_labels = perform_kmeans(
        X,
        n_clusters
    )

    hierarchical_model, hierarchical_labels = perform_hierarchical(
        X,
        n_clusters
    )

    gmm_model, gmm_labels = perform_gmm(
        X,
        n_clusters
    )

    dbscan_model, dbscan_labels = perform_dbscan(
        X
    )

    return {

        "kmeans": (
            kmeans_model,
            kmeans_labels
        ),

        "hierarchical": (
            hierarchical_model,
            hierarchical_labels
        ),

        "gmm": (
            gmm_model,
            gmm_labels
        ),

        "dbscan": (
            dbscan_model,
            dbscan_labels
        )
    }


# Evaluate

def evaluate_models(
        X,
        results
):

    scores = []

    for name, (model, labels) in results.items():

        n_found = len(set(labels)) - (1 if -1 in labels else 0)

        if n_found < 2:
            scores.append({
                "model": name,
                "n_clusters_found": n_found,
                "silhouette_score": None,
                "davies_bouldin_score": None
            })
            continue

        scores.append({
            "model": name,
            "n_clusters_found": n_found,
            "silhouette_score": silhouette_score(X, labels),
            "davies_bouldin_score": davies_bouldin_score(X, labels)
        })

    scores_df = pd.DataFrame(scores).set_index("model")

    return scores_df


def select_best_model(
        results,
        scores_df,
        model_path=None
):

    valid_scores = scores_df.dropna(subset=["silhouette_score"])
    best_name = valid_scores["silhouette_score"].idxmax()
    best_model, best_labels = results[best_name]

    if model_path:
        joblib.dump(best_model, model_path)

    return best_name, best_model, best_labels


# PCA

def apply_pca(
        X,
        n_components=2
):

    pca = PCA(
        n_components=n_components,
        random_state=42
    )

    X_pca = pca.fit_transform(X)

    return X_pca, pca


def plot_clusters_pca(
        X,
        labels,
        title="Cluster Visualization (PCA)",
        save_path=None
):

    X_pca, _ = apply_pca(X, n_components=2)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="tab10",
        s=10
    )

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label="Cluster")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.close()


# Cluster interpretation

def profile_clusters(
        df_raw,
        labels,
        save_path=None
):

    df_raw = df_raw.copy()
    df_raw["CLUSTER"] = labels

    profile = df_raw.groupby("CLUSTER").mean(numeric_only=True)

    if save_path:
        profile.to_csv(save_path)

    return profile


def plot_cluster_sizes(
        labels,
        save_path=None
):

    counts = pd.Series(labels).value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values, color="#4C72B0")
    plt.title("Customers per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of customers")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.close()
