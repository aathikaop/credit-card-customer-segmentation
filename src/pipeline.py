# Import Libraries

import os
import pandas as pd
import joblib

from src.feature_engineering import add_features
from src.preprocessing import create_preprocessing_pipeline, drop_cols

from src.clustering import (
    find_optimal_k,
    plot_elbow_and_silhouette,
    run_clustering,
    evaluate_models,
    select_best_model,
    plot_clusters_pca,
    plot_cluster_sizes,
    profile_clusters,
    N_CLUSTERS
)


# File paths

RAW_PATH = "data/raw/CC GENERAL.csv"
PROCESSED_PATH = "data/processed/cc_general_processed.csv"

MODEL_PATH = "outputs/models/best_clustering_model.pkl"
PREPROCESSING_PATH = "outputs/models/preprocessing_pipeline.pkl"
CLUSTERS_PATH = "outputs/clusters/customer_clusters.csv"
FINAL_DATASET_PATH = "outputs/clusters/dataset_with_clusters.csv"

ELBOW_PLOT_PATH = "outputs/reports/elbow_silhouette.png"
COMPARISON_PATH = "outputs/reports/model_comparison.csv"
PCA_PLOT_PATH = "outputs/reports/cluster_pca_plot.png"
SIZES_PLOT_PATH = "outputs/reports/cluster_sizes.png"
PROFILE_PATH = "outputs/reports/cluster_profile.csv"


def run_pipeline():


    for path in [
        PROCESSED_PATH,
        MODEL_PATH,
        PREPROCESSING_PATH,
        CLUSTERS_PATH,
        FINAL_DATASET_PATH,
        ELBOW_PLOT_PATH,
        COMPARISON_PATH,
        PCA_PLOT_PATH,
        SIZES_PLOT_PATH,
        PROFILE_PATH,
    ]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # load raw data

    df_raw = pd.read_csv(RAW_PATH)
    ids = df_raw["CUST_ID"]

    # impute missing values.

    df_raw["MINIMUM_PAYMENTS"] = df_raw["MINIMUM_PAYMENTS"].fillna(
        df_raw["MINIMUM_PAYMENTS"].median()
    )
    df_raw["CREDIT_LIMIT"] = df_raw["CREDIT_LIMIT"].fillna(
        df_raw["CREDIT_LIMIT"].median()
    )

    # feature engineering

    df_features = add_features(df_raw)

    # preprocessing

    preprocessing_pipeline = create_preprocessing_pipeline()
    X_processed = preprocessing_pipeline.fit_transform(df_features)

    joblib.dump(preprocessing_pipeline, PREPROCESSING_PATH)

     # convert to Dataframe
    feature_names = df_features.drop(columns=drop_cols).columns
    X = pd.DataFrame(X_processed, columns=feature_names)
    X.to_csv(PROCESSED_PATH, index=False)

    print(f"Preprocessing done. Shape: {X.shape}")


    # find optimal k for KMeans and save plot
    k_values, inertia, silhouette = find_optimal_k(X)

    plot_elbow_and_silhouette(
        k_values,
        inertia,
        silhouette,
        save_path=ELBOW_PLOT_PATH
    )

    print(f"Elbow/silhouette plot saved. Using N_CLUSTERS = {N_CLUSTERS}")


    # Run every clustering algorithm
    results = run_clustering(X, n_clusters=N_CLUSTERS)

    # Score and compare all models
    scores_df = evaluate_models(X, results)
    scores_df.to_csv(COMPARISON_PATH)

    print("Model comparison:")
    print(scores_df)

    # select the best model and save
    best_name, best_model, best_labels = select_best_model(
        results,
        scores_df,
        model_path=MODEL_PATH
    )

    print(f"Best model: {best_name}")


    # visualize the best model's clusters
    plot_clusters_pca(
        X,
        best_labels,
        title=f"Cluster Visualization - {best_name} (PCA)",
        save_path=PCA_PLOT_PATH
    )

    plot_cluster_sizes(
        best_labels,
        save_path=SIZES_PLOT_PATH
    )

    # profile clusters
    profile = profile_clusters(
        df_raw.drop(columns=["CUST_ID"]),
        best_labels,
        save_path=PROFILE_PATH
    )

    print("Cluster profile:")
    print(profile)

    # save outputs

    ##customer and their labels
    result = pd.DataFrame({
        "CUST_ID": ids,
        "CLUSTER": best_labels
    })
    result.to_csv(CLUSTERS_PATH, index=False)

    ## raw data and cluster label together
    final_dataset = df_raw.copy()
    final_dataset["CLUSTER"] = best_labels
    final_dataset.to_csv(FINAL_DATASET_PATH, index=False)

    print(f"Pipeline complete. Final dataset saved to {FINAL_DATASET_PATH}")

    return final_dataset