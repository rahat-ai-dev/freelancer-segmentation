# 🌍 Global Freelancer Segmentation

**Unsupervised customer segmentation applied to the global gig economy** — an end-to-end machine learning project (data cleaning → feature engineering → KMeans clustering → interactive Streamlit dashboard).

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KMeans%20%7C%20PCA-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

🔗 **Live app:** [freelancer-segmentation-cyjocdsmnxlen5b94wquda.streamlit.app](https://freelancer-segmentation-cyjocdsmnxlen5b94wquda.streamlit.app/)
📦 **Repository:** [github.com/rahat-ai-dev/freelancer-segmentation](https://github.com/rahat-ai-dev/freelancer-segmentation)

---

## The idea

Most beginner segmentation tutorials cluster mall shoppers or credit-card
customers — the same dataset, over and over. This project applies the same
unsupervised learning technique (KMeans + PCA) to a different, deliberately
chosen problem: **segmenting freelancers on Fiverr, Upwork, Freelancer.com,
PeoplePerHour, and Toptal by how they actually work and earn**, not by
where they live.

That framing was chosen on purpose. It doesn't just classify Bangladeshi
freelancers or only international ones — it groups *behavior*, so the same
segment can contain someone in Dhaka and someone in Toronto. That's what
makes the project relatable to a global audience while still speaking
directly to a huge, real community: Bangladesh is consistently one of the
top freelancer-supplying countries on platforms like Upwork.

**The segments this project discovers are business-relevant, not just
descriptive** — e.g. *Premium Specialists*, *Rising High-Earners*, *Trusted
Steady Performers*, *New & Emerging Talent* — the kind of grouping a
platform's growth team, or a freelancer trying to benchmark themselves,
could actually act on.

## Dataset

**[Freelancer Earnings & Job Trends](https://www.kaggle.com/datasets/shohinurpervezshohan/freelancer-earnings-and-job-trends)** — a real, publicly downloadable Kaggle dataset (not synthetic data). It contains per-freelancer records with:

`Freelancer_ID`, `Job_Category`, `Platform`, `Experience_Level`, `Client_Region`, `Payment_Method`, `Job_Completed`, `Earnings_USD`, `Hourly_Rate`, `Job_Success_Rate`, `Client_Rating`, `Job_Duration_Days`, `Project_Type`, `Rehire_Rate`, `Marketing_Spend`

The raw CSV is **not** committed to this repository (see [Getting the data](#1-get-the-dataset)) — download it yourself and drop it into `data/raw/`.

## What makes this an "unsupervised" project

There is no target label (no "correct" segment) anywhere in the data or the
pipeline. The model:

1. Cleans and engineers behavioral features (earnings per job, marketing
   efficiency, a blended client-trust index, etc.) — see
   [`src/feature_engineering.py`](src/feature_engineering.py).
2. Scales them with `StandardScaler`.
3. Sweeps `k = 2..10` and picks the number of clusters with the **best
   silhouette score** — not a guessed or hard-coded `k`.
4. Fits `KMeans` and reduces to 2D with `PCA` purely for visualization.
5. **Auto-names** each cluster from its own statistics (rank on earnings,
   client trust, and experience) so labels always reflect the current
   model instead of being manually hard-coded.

All of that logic is reusable — the same functions run in the training
script, the exploratory notebook, and the live Streamlit app when a user
uploads their own CSV.

## Project structure

```
freelancer-segmentation/
├── README.md
├── requirements.txt
├── train_pipeline.py              # run this first — builds the model
├── data/
│   ├── raw/                       # put the Kaggle CSV here (not committed)
│   └── processed/                 # outputs of train_pipeline.py
├── models/                        # saved scaler + PCA + KMeans + cluster names
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── src/
│   ├── data_preprocessing.py      # load + clean the raw CSV
│   ├── feature_engineering.py     # behavioral feature engineering
│   ├── clustering.py              # k-selection, KMeans, PCA, cluster naming
│   └── utils.py                   # shared Plotly chart builders
└── app/
    └── streamlit_app.py           # the interactive dashboard
```

## Getting started

### 0. Set up the environment

```bash
git clone https://github.com/rahat-ai-dev/freelancer-segmentation.git
cd freelancer-segmentation
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Get the dataset

Option A — manual download:
1. Go to the [dataset page on Kaggle](https://www.kaggle.com/datasets/shohinurpervezshohan/freelancer-earnings-and-job-trends).
2. Click **Download**.
3. Place the CSV in `data/raw/` and rename it to
   `freelancer_earnings_and_job_trends.csv` (or pass `--input` in step 2).

Option B — Kaggle API:
```bash
pip install kaggle
# with your kaggle.json API token in ~/.kaggle/
kaggle datasets download -d shohinurpervezshohan/freelancer-earnings-and-job-trends -p data/raw --unzip
```

### 2. Train the model

```bash
python train_pipeline.py
```

This cleans the data, engineers features, automatically picks the best
number of clusters, and saves everything the app needs into
`data/processed/` and `models/`.

Optional flags:
```bash
python train_pipeline.py --input data/raw/your_file.csv --k 5
```

### 3. Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

### 4. Explore the EDA notebook (optional)

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

## Dashboard features

| Page | What it shows |
|---|---|
| **Overview** | Headline metrics and a plain-English explanation of the methodology |
| **Explore the Segments** | PCA scatter plot, normalized radar comparison, and a full profile table (with CSV download) |
| **Global Client Map** | Which client regions and platforms each segment actually serves — the cross-border view |
| **Find My Segment** | A form where any visitor enters their own stats and gets matched to a segment live |
| **About This Project** | Methodology and tech-stack summary |

You can also upload your own CSV (same schema) from the sidebar and the
app will clean it, engineer features, and cluster it live — no need to
retrain from the command line.

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub. **Important:** run `train_pipeline.py`
   locally first and commit `data/processed/*.csv` and `models/*.pkl` —
   Streamlit Cloud only has what's in the repo, so it can't train the
   model itself unless you also commit the raw dataset (not recommended
   for size/licensing reasons).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, select this repository, and set:
   - **Main file path:** `app/streamlit_app.py`
   - **Python version:** 3.10+
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically.
5. Share the generated `*.streamlit.app` URL — it works the same for a
   viewer in Dhaka or in Toronto.

## Tech stack

- **Data processing:** pandas, NumPy
- **Modeling:** scikit-learn (`StandardScaler`, `PCA`, `KMeans`, `silhouette_score`)
- **Visualization:** Plotly, Matplotlib/Seaborn (notebook)
- **App:** Streamlit
- **Persistence:** joblib

## Author

Built by **Rahat** ([GitHub: rahat-ai-dev](https://github.com/rahat-ai-dev) · [Hugging Face: rahat-dev](https://huggingface.co/rahat-dev)) as part of an independent AI/ML portfolio.

- Repository: [github.com/rahat-ai-dev/freelancer-segmentation](https://github.com/rahat-ai-dev/freelancer-segmentation)
- Live app: [freelancer-segmentation-cyjocdsmnxlen5b94wquda.streamlit.app](https://freelancer-segmentation-cyjocdsmnxlen5b94wquda.streamlit.app/)

## License

MIT — free to use, adapt, and build on. If this helped you, a star on the repo is appreciated.