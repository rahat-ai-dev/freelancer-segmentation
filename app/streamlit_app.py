"""
Streamlit app - Global Freelancer Segmentation
================================================
An unsupervised-learning dashboard that segments freelancers from the
Kaggle "Freelancer Earnings & Job Trends" dataset into behavioral
groups (KMeans + PCA), then lets any visitor - a client in London, a
freelancer in Dhaka, a recruiter in Dubai - explore the segments and
find out which one their own profile would fall into.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `src` importable regardless of the working directory Streamlit is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.clustering import (  # noqa: E402
    fit_pipeline,
    load_artifacts,
    name_clusters,
    predict_cluster,
    profile_clusters,
)
from src.data_preprocessing import clean_data  # noqa: E402
from src.feature_engineering import build_feature_matrix  # noqa: E402
from src.utils import (  # noqa: E402
    cluster_size_figure,
    pca_scatter_figure,
    radar_figure,
    region_by_segment_figure,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SEGMENTED_CSV = PROCESSED_DIR / "segmented_freelancers.csv"
CLUSTER_PROFILE_CSV = PROCESSED_DIR / "cluster_profile.csv"
ARTIFACTS_PKL = MODELS_DIR / "clustering_artifacts.pkl"

st.set_page_config(
    page_title="Global Freelancer Segmentation",
    page_icon="🌍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_precomputed():
    df = pd.read_csv(SEGMENTED_CSV)
    profile = pd.read_csv(CLUSTER_PROFILE_CSV, index_col=0)
    profile.index = profile.index.astype(int)
    return df, profile


@st.cache_resource(show_spinner=False)
def load_model():
    return load_artifacts(str(ARTIFACTS_PKL))


@st.cache_data(show_spinner="Cleaning data and running KMeans on your upload ...")
def run_pipeline_on_upload(raw_bytes: bytes):
    import io
    from src.clustering import evaluate_k_range
    from sklearn.preprocessing import StandardScaler

    raw_df = pd.read_csv(io.BytesIO(raw_bytes))
    cleaned = clean_data(raw_df)
    featured, feature_names = build_feature_matrix(cleaned)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(featured[feature_names].values)
    k_scores = evaluate_k_range(X_scaled, k_min=2, k_max=8)
    best_k = int(k_scores.loc[k_scores["silhouette"].idxmax(), "k"])

    segmented, artifacts = fit_pipeline(featured, feature_names, k=best_k)
    profile = profile_clusters(segmented, feature_names)
    names = name_clusters(profile)
    segmented["segment_name"] = segmented["cluster"].map(names)
    return segmented, profile, names, artifacts


has_precomputed = SEGMENTED_CSV.exists() and CLUSTER_PROFILE_CSV.exists() and ARTIFACTS_PKL.exists()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🌍 Freelancer Segmentation")
st.sidebar.caption("Unsupervised learning · KMeans + PCA")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Explore the Segments", "Global Client Map", "Find My Segment", "About This Project"],
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Optional: upload your own export of the dataset (CSV)",
    type="csv",
    help="Uses the same schema as the Kaggle 'Freelancer Earnings & Job Trends' dataset. "
         "If you don't upload anything, the app uses the pre-trained model shipped with it.",
)

if uploaded_file is not None:
    segmented_df, cluster_profile, cluster_names, artifacts = run_pipeline_on_upload(
        uploaded_file.getvalue()
    )
    st.sidebar.success(f"Trained live on your upload · k={artifacts.k}")
elif has_precomputed:
    segmented_df, cluster_profile = load_precomputed()
    artifacts = load_model()
    import json
    with open(MODELS_DIR / "cluster_names.json") as f:
        cluster_names = {int(k): v for k, v in json.load(f).items()}
else:
    st.error(
        "No trained model found yet.\n\n"
        "Run `python train_pipeline.py` first (after placing the Kaggle CSV in "
        "`data/raw/`), or upload a CSV in the sidebar to train on the fly."
    )
    st.stop()

feature_names = artifacts.feature_names

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

if page == "Overview":
    st.title("Global Freelancer Segmentation")
    st.markdown(
        """
The gig economy no longer has a single home base. A designer in Dhaka,
a developer in Lagos, and a copywriter in Manila can all be logged
into the same platform, competing for - and winning - the same
international clients. **Traditional customer segmentation projects
group shoppers by what they buy. This project instead groups
freelancers by how they *earn* - their pricing behavior, delivery
reliability, and client trust - to answer a genuinely cross-border
question: *what kind of freelancer are you, regardless of which
country you work from?***
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Freelancers analyzed", f"{len(segmented_df):,}")
    col2.metric("Segments discovered", f"{segmented_df['cluster'].nunique()}")
    if "Platform" in segmented_df.columns:
        col3.metric("Platforms covered", f"{segmented_df['Platform'].nunique()}")
    if "Client_Region" in segmented_df.columns:
        col4.metric("Client regions covered", f"{segmented_df['Client_Region'].nunique()}")

    st.plotly_chart(cluster_size_figure(segmented_df, cluster_names), use_container_width=True)

    st.markdown("### How the segments were built")
    st.markdown(
        f"""
1. **Clean** the raw Kaggle data (duplicates, missing values, impossible values, outlier capping).
2. **Engineer** {len(feature_names)} behavioral features - earnings per job, marketing efficiency,
   a blended client-trust index, and more - instead of clustering on raw earnings alone.
3. **Scale** every feature (StandardScaler) so high-magnitude columns like earnings don't
   dominate the distance metric.
4. **Reduce** to 2 dimensions with PCA for visualization.
5. **Cluster** with KMeans, choosing *k* by sweeping k=2..10 and picking the highest
   silhouette score - not an arbitrary guess.
6. **Name** each cluster automatically from its own feature averages, so labels always
   reflect the current model rather than a hard-coded guess.
        """
    )

# ---------------------------------------------------------------------------
# Page: Explore the Segments
# ---------------------------------------------------------------------------

elif page == "Explore the Segments":
    st.title("Explore the Segments")

    tab1, tab2, tab3 = st.tabs(["Segment Map (PCA)", "Segment Profiles (Radar)", "Segment Table"])

    with tab1:
        st.plotly_chart(pca_scatter_figure(segmented_df, cluster_names), use_container_width=True)
        st.caption(
            "Each point is one freelancer, projected from the full feature space down to "
            "2 dimensions with PCA. Points that cluster tightly together behave similarly "
            "across earnings, reliability, and pricing."
        )

    with tab2:
        st.plotly_chart(radar_figure(cluster_profile, cluster_names), use_container_width=True)
        st.caption(
            "Each feature is normalized 0-1 across segments, so the shapes are directly "
            "comparable - a segment that bulges out on 'client_trust_index' is more "
            "reliable than average, regardless of its absolute earnings."
        )

    with tab3:
        display_profile = cluster_profile.copy()
        display_profile.index = [cluster_names.get(i, i) for i in display_profile.index]
        st.dataframe(display_profile, use_container_width=True)

        st.download_button(
            "Download segmented dataset (CSV)",
            data=segmented_df.to_csv(index=False).encode("utf-8"),
            file_name="segmented_freelancers.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("What each segment means")
    for cid, name in sorted(cluster_names.items(), key=lambda x: x[0]):
        size = (segmented_df["cluster"] == cid).sum()
        pct = size / len(segmented_df) * 100
        with st.expander(f"{name}  ·  {size:,} freelancers ({pct:.1f}%)"):
            row = cluster_profile.loc[cid]
            st.write(row.to_frame(name="Average value"))

# ---------------------------------------------------------------------------
# Page: Global Client Map
# ---------------------------------------------------------------------------

elif page == "Global Client Map":
    st.title("Global Client Map")
    st.markdown(
        "This is the part of the project that makes it genuinely international: "
        "which client regions each segment actually works with. A freelancer's segment "
        "isn't tied to their own country - it's tied to their earning behavior, and this "
        "view shows how those behaviors are distributed across the clients they serve."
    )
    fig = region_by_segment_figure(segmented_df, cluster_names)
    if fig.data:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("This dataset doesn't include a 'Client_Region' column, so this view is unavailable.")

    if "Platform" in segmented_df.columns:
        st.subheader("Segment composition by platform")
        platform_tab = (
            segmented_df.assign(Segment=segmented_df["cluster"].map(cluster_names))
            .groupby(["Platform", "Segment"]).size().unstack(fill_value=0)
        )
        st.bar_chart(platform_tab)

# ---------------------------------------------------------------------------
# Page: Find My Segment
# ---------------------------------------------------------------------------

elif page == "Find My Segment":
    st.title("Find My Segment")
    st.markdown(
        "Enter your own (or a hypothetical) freelancer profile and the trained model "
        "will tell you which of the discovered segments you're closest to."
    )

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        earnings = c1.number_input("Total earnings (USD)", min_value=0.0, value=2500.0, step=100.0)
        hourly_rate = c1.number_input("Hourly rate (USD)", min_value=0.0, value=25.0, step=1.0)
        jobs_completed = c1.number_input("Jobs completed", min_value=1, value=20, step=1)

        success_rate = c2.slider("Job success rate (%)", 0, 100, 90)
        client_rating = c2.slider("Average client rating (1-5)", 1.0, 5.0, 4.5, step=0.1)
        rehire_rate = c2.slider("Rehire rate (%)", 0, 100, 40)

        job_duration = c3.number_input("Typical job duration (days)", min_value=1, value=10, step=1)
        marketing_spend = c3.number_input("Marketing / bidding spend (USD)", min_value=0.0, value=50.0, step=10.0)
        experience_level = c3.selectbox("Experience level", ["Beginner", "Intermediate", "Expert"], index=1)

        submitted = st.form_submit_button("Predict my segment", use_container_width=True)

    if submitted:
        exp_map = {"Beginner": 1, "Intermediate": 2, "Expert": 3}
        trust_index = (success_rate + (client_rating / 5 * 100) + rehire_rate) / 3
        feature_row = {
            "Earnings_USD": earnings,
            "Hourly_Rate": hourly_rate,
            "Job_Completed": jobs_completed,
            "Job_Success_Rate": success_rate,
            "Client_Rating": client_rating,
            "Rehire_Rate": rehire_rate,
            "Job_Duration_Days": job_duration,
            "Marketing_Spend": marketing_spend,
            "earnings_per_job": earnings / jobs_completed,
            "marketing_efficiency": earnings / (marketing_spend + 1),
            "client_trust_index": trust_index,
            "experience_score": exp_map[experience_level],
        }
        # Only pass along features the model was actually trained with.
        usable = {f: feature_row[f] for f in feature_names if f in feature_row}
        predicted_cluster = predict_cluster(artifacts, usable)
        segment_label = cluster_names.get(predicted_cluster, f"Cluster {predicted_cluster}")

        st.success(f"You match the **{segment_label}** segment.")
        avg_row = cluster_profile.loc[predicted_cluster]
        st.write("How your inputs compare to this segment's average:")
        compare_df = pd.DataFrame({
            "You": pd.Series(usable),
            f"{segment_label} average": avg_row,
        }).dropna()
        st.dataframe(compare_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------

elif page == "About This Project":
    st.title("About This Project")
    st.markdown(
        """
**Project:** Global Freelancer Segmentation
**Type:** Unsupervised Machine Learning (KMeans clustering + PCA)
**Dataset:** [Freelancer Earnings & Job Trends](https://www.kaggle.com/datasets/shohinurpervezshohan/freelancer-earnings-and-job-trends) (Kaggle, real-world data - not synthetic)

### Why this niche
Most beginner segmentation projects cluster mall shoppers or credit-card
customers. This project instead segments the global freelance workforce -
Fiverr, Upwork, Freelancer.com, PeoplePerHour, and Toptal - by behavior
rather than geography. That framing was chosen deliberately: it connects
naturally with an audience of freelancers and clients in both Bangladesh
and abroad, since the segments describe *how someone works*, not *where
they're from*.

### Tech stack
- **Data processing:** pandas, NumPy
- **Modeling:** scikit-learn (StandardScaler, PCA, KMeans, silhouette score)
- **Visualization:** Plotly
- **App / deployment:** Streamlit, Streamlit Community Cloud

### Project structure
```
freelancer-segmentation/
├── data/raw/            # place the Kaggle CSV here
├── data/processed/      # outputs of train_pipeline.py
├── models/              # saved scaler + PCA + KMeans + cluster names
├── src/                 # reusable pipeline code
├── app/streamlit_app.py # this dashboard
└── train_pipeline.py    # run this first
```
        """
    )
