"""
utils.py
---------
Small shared helpers (mostly Plotly figure builders) used by both the
training pipeline and the Streamlit app, so chart styling stays
consistent in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CLUSTER_COLOR_SEQUENCE = px.colors.qualitative.Set2


def elbow_figure(k_scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_scores["k"], y=k_scores["inertia"],
        mode="lines+markers", name="Inertia",
        line=dict(color="#4C78A8", width=3),
    ))
    fig.update_layout(
        title="Elbow Method - Inertia vs. Number of Clusters",
        xaxis_title="Number of clusters (k)",
        yaxis_title="Inertia (within-cluster sum of squares)",
        template="plotly_white",
    )
    return fig


def silhouette_figure(k_scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_scores["k"], y=k_scores["silhouette"],
        mode="lines+markers", name="Silhouette score",
        line=dict(color="#F58518", width=3),
    ))
    fig.update_layout(
        title="Silhouette Score vs. Number of Clusters",
        xaxis_title="Number of clusters (k)",
        yaxis_title="Silhouette score (higher = better separated)",
        template="plotly_white",
    )
    return fig


def pca_scatter_figure(df: pd.DataFrame, cluster_names: dict[int, str]) -> go.Figure:
    plot_df = df.copy()
    plot_df["Segment"] = plot_df["cluster"].map(cluster_names)
    fig = px.scatter(
        plot_df, x="pca_1", y="pca_2", color="Segment",
        color_discrete_sequence=CLUSTER_COLOR_SEQUENCE,
        hover_data={c: True for c in ["Platform", "Client_Region", "Experience_Level"]
                    if c in plot_df.columns},
        title="Freelancer Segments (PCA-Reduced View)",
        opacity=0.75,
    )
    fig.update_layout(template="plotly_white", legend_title_text="Segment")
    return fig


def radar_figure(cluster_profile: pd.DataFrame, cluster_names: dict[int, str]) -> go.Figure:
    """Normalizes each feature to 0-1 across clusters so the radar shape is comparable."""
    norm = (cluster_profile - cluster_profile.min()) / (
        cluster_profile.max() - cluster_profile.min() + 1e-9
    )
    fig = go.Figure()
    for cid in norm.index:
        fig.add_trace(go.Scatterpolar(
            r=norm.loc[cid].values,
            theta=norm.columns,
            fill="toself",
            name=cluster_names.get(cid, f"Cluster {cid}"),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Segment Profiles (Normalized Feature Comparison)",
        template="plotly_white",
    )
    return fig


def cluster_size_figure(df: pd.DataFrame, cluster_names: dict[int, str]) -> go.Figure:
    counts = df["cluster"].map(cluster_names).value_counts().reset_index()
    counts.columns = ["Segment", "Count"]
    fig = px.bar(
        counts, x="Segment", y="Count", color="Segment",
        color_discrete_sequence=CLUSTER_COLOR_SEQUENCE,
        title="Freelancers per Segment",
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def region_by_segment_figure(df: pd.DataFrame, cluster_names: dict[int, str]) -> go.Figure:
    if "Client_Region" not in df.columns:
        return go.Figure()
    plot_df = df.copy()
    plot_df["Segment"] = plot_df["cluster"].map(cluster_names)
    grouped = (
        plot_df.groupby(["Client_Region", "Segment"])
        .size()
        .reset_index(name="Count")
    )
    fig = px.bar(
        grouped, x="Client_Region", y="Count", color="Segment",
        color_discrete_sequence=CLUSTER_COLOR_SEQUENCE,
        title="Client Region Composition by Segment",
        barmode="stack",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Client Region")
    return fig
