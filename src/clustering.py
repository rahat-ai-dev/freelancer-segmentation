"""
clustering.py
---------------
The unsupervised-learning core of the project.

Pipeline:
    raw numeric features -> StandardScaler -> PCA (for visualization
    and optional dimensionality reduction) -> KMeans

Also includes:
    - k selection via the elbow method + silhouette score
    - a rule-based cluster-naming function so the app can show
      "High-Earning Specialists" instead of "Cluster 2"
    - joblib persistence helpers
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringArtifacts:
    scaler: StandardScaler
    pca: PCA
    kmeans: KMeans
    feature_names: list[str]
    k: int


def evaluate_k_range(
    X_scaled: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int = 42
) -> pd.DataFrame:
    """
    Fits KMeans for every k in [k_min, k_max] and records inertia
    (for the elbow plot) and silhouette score (for a more objective
    pick). Returns a small DataFrame the Streamlit app can chart
    directly.
    """
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if k > 1 else np.nan
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def fit_pipeline(
    df: pd.DataFrame,
    feature_names: list[str],
    k: int,
    pca_components: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, ClusteringArtifacts]:
    """
    Fits scaler -> PCA -> KMeans on df[feature_names] and returns the
    original dataframe with `cluster` and PCA coordinate columns
    attached, plus the fitted artifacts for reuse in the app.
    """
    X = df[feature_names].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=pca_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    out = df.copy()
    out["cluster"] = labels
    for i in range(pca_components):
        out[f"pca_{i + 1}"] = X_pca[:, i]

    artifacts = ClusteringArtifacts(
        scaler=scaler, pca=pca, kmeans=kmeans, feature_names=feature_names, k=k
    )
    return out, artifacts


def predict_cluster(artifacts: ClusteringArtifacts, feature_row: dict) -> int:
    """Predicts the cluster for a single new freelancer profile (dict of feature -> value)."""
    x = np.array([[feature_row[f] for f in artifacts.feature_names]])
    x_scaled = artifacts.scaler.transform(x)
    return int(artifacts.kmeans.predict(x_scaled)[0])


def profile_clusters(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Mean feature value per cluster - the basis for naming and the radar chart."""
    return df.groupby("cluster")[feature_names].mean().round(2)


def name_clusters(cluster_profile: pd.DataFrame) -> dict[int, str]:
    """
    Rule-based, data-driven cluster naming. Ranks clusters on earnings
    and client-trust percentiles and combines them into a short,
    business-friendly label. This runs fresh every time the model is
    retrained, so names always reflect the *current* clustering rather
    than being hard-coded guesses.
    """
    names = {}
    earn_col = "Earnings_USD" if "Earnings_USD" in cluster_profile.columns else None
    trust_col = "client_trust_index" if "client_trust_index" in cluster_profile.columns else None
    exp_col = "experience_score" if "experience_score" in cluster_profile.columns else None

    earn_rank = cluster_profile[earn_col].rank(pct=True) if earn_col else None
    trust_rank = cluster_profile[trust_col].rank(pct=True) if trust_col else None
    exp_rank = cluster_profile[exp_col].rank(pct=True) if exp_col else None

    for cid in cluster_profile.index:
        e = earn_rank.loc[cid] if earn_rank is not None else 0.5
        t = trust_rank.loc[cid] if trust_rank is not None else 0.5
        x = exp_rank.loc[cid] if exp_rank is not None else 0.5

        if e >= 0.75 and t >= 0.6:
            label = "Premium Specialists"
        elif e >= 0.6 and x < 0.5:
            label = "Rising High-Earners"
        elif t >= 0.75 and e < 0.6:
            label = "Trusted Steady Performers"
        elif e < 0.4 and x < 0.4:
            label = "New & Emerging Talent"
        elif e < 0.4 and t < 0.4:
            label = "At-Risk / Needs Support"
        else:
            label = "Balanced Mid-Tier Freelancers"
        names[cid] = label
    return names


def save_artifacts(artifacts: ClusteringArtifacts, out_dir: str) -> None:
    joblib.dump(artifacts, f"{out_dir}/clustering_artifacts.pkl")


def load_artifacts(path: str) -> ClusteringArtifacts:
    return joblib.load(path)
