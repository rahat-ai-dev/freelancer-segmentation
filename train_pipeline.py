"""
train_pipeline.py
--------------------
Run this once (or whenever the raw dataset changes) to:

    1. Load + clean the raw Kaggle CSV
    2. Engineer clustering features
    3. Sweep k = 2..10 and pick the k with the best silhouette score
    4. Fit the final scaler -> PCA -> KMeans pipeline
    5. Save the segmented dataset and the fitted model artifacts

Usage:
    python train_pipeline.py
    python train_pipeline.py --input data/raw/freelancer_earnings_and_job_trends.csv --k 5

The Streamlit app loads whatever this script produces in
data/processed/ and models/, so re-run this any time you want the
dashboard to reflect a fresh dataset or a manually chosen k.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.clustering import (
    evaluate_k_range,
    fit_pipeline,
    name_clusters,
    profile_clusters,
    save_artifacts,
)
from src.data_preprocessing import load_and_clean
from src.feature_engineering import build_feature_matrix
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "data/raw/freelancer_earnings_and_job_trends.csv"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the freelancer segmentation model.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                         help="Path to the raw Kaggle CSV.")
    parser.add_argument("--k", type=int, default=None,
                         help="Force a specific number of clusters. If omitted, "
                              "the best silhouette score across k=2..10 is used.")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading and cleaning raw data from {args.input} ...")
    df = load_and_clean(args.input)

    print("[2/5] Engineering features ...")
    df, feature_names = build_feature_matrix(df)
    print(f"       Using {len(feature_names)} features: {feature_names}")

    print(f"[3/5] Sweeping k = {args.k_min}..{args.k_max} to score cluster quality ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_names].values)
    k_scores = evaluate_k_range(X_scaled, k_min=args.k_min, k_max=args.k_max)
    k_scores.to_csv(PROCESSED_DIR / "k_selection_scores.csv", index=False)

    if args.k:
        best_k = args.k
        print(f"Using user-specified k = {best_k}")
    else:
        best_k = int(k_scores.loc[k_scores["silhouette"].idxmax(), "k"])
        print(f"       Best k by silhouette score = {best_k}")
        print(k_scores.to_string(index=False))

    print(f"[4/5] Fitting final pipeline with k = {best_k} ...")
    segmented_df, artifacts = fit_pipeline(df, feature_names, k=best_k)

    cluster_profile = profile_clusters(segmented_df, feature_names)
    cluster_names = name_clusters(cluster_profile)
    segmented_df["segment_name"] = segmented_df["cluster"].map(cluster_names)

    print("[5/5] Saving outputs ...")
    segmented_df.to_csv(PROCESSED_DIR / "segmented_freelancers.csv", index=False)
    cluster_profile.to_csv(PROCESSED_DIR / "cluster_profile.csv")
    save_artifacts(artifacts, str(MODELS_DIR))

    import json
    with open(MODELS_DIR / "cluster_names.json", "w") as f:
        json.dump({str(k): v for k, v in cluster_names.items()}, f, indent=2)

    print("\nDone. Segment sizes:")
    print(segmented_df["segment_name"].value_counts().to_string())
    print(f"\nProcessed data  -> {PROCESSED_DIR}")
    print(f"Model artifacts -> {MODELS_DIR}")
    print("\nYou can now run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
