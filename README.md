# Predicting Video Game Commercial Success on Steam
### An End-to-End Supervised Learning Pipeline with Temporal Validation and SHAP Interpretability

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-green)](https://www.mit.edu/~amini/LICENSE.md)

---

## Research Question

> *What game characteristics, observable at the time of release, are most predictive of long term commercial success on Steam — and does this relationship vary by genre?*

Most existing analyses of Steam data treat success prediction as a straightforward regression task. This project addresses two methodological gaps commonly overlooked in similar work: (1) the use of random train/test splits that leak future information across a time-ordered dataset, and (2) the absence of post-hoc interpretability that would allow actionable conclusions beyond leaderboard metrics.

---

## Project Status

| Stage | Status | Notes |
|---|---|---|
| EDA — round 1 (raw data) | ✅ Completed | See `notebooks/01_eda_raw.ipynb` |
| Cleaning — round 1 (structural) | ✅ Completed | See `notebooks/02_cleaning_initial.ipynb` |
| EDA — round 2 (patterns & target) | 🔄 In Progress | See `notebooks/03_eda_final.ipynb` |
| Cleaning — round 2 (final prep) | ⬜ Planned | See `notebooks/04_cleaning_final.ipynb` |
| Feature engineering | ⬜ Planned | Time, price, genre, tag features |
| Baseline model (Linear Regression) | ⬜ Planned | |
| Tree models (Random Forest, XGBoost) | ⬜ Planned | |
| SHAP analysis & error analysis | ⬜ Planned | |
| MLflow experiment tracking | ⬜ Planned | |
| Streamlit deployment | ⬜ Planned | |

---

## Dataset

**Source:** [Steam Games Dataset — Kaggle](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)

~122,000 raw games including release date, genre, tags, price, review count, and review sentiment ratio.

> `games_clean.r1.csv` is not included in this repo due to file size.  
> To reproduce it, run `notebooks/02_cleaning_initial.ipynb`.

**Target variable:** A composite success score defined as:

```
success_score = log1p(review_count) × wilson_lower_bound(positive, negative)
```

This captures both reach (total reviews as a proxy for sales volume) and quality (review confidence), reflecting each game's cumulative performance up to the dataset snapshot date. The log transform addresses the strong right skew in review counts. The Wilson lower bound replaces a raw positive ratio, penalising games with few reviews where the observed ratio is statistically unreliable. The choice of this formulation over alternatives (raw review count, rating alone) is discussed in `notebooks/03_eda_final.ipynb`.

---

## Data Pipeline

This project uses an **iterative two-pass approach** to EDA and cleaning, rather than a single linear pass. This reflects how data issues are discovered in practice: you cannot know what to clean until you have looked at the data, and you cannot trust patterns in the data until it is clean enough to analyse.

### Round 1 — understand the raw data, fix structural problems

**EDA round 1** (`notebooks/01_eda_raw.ipynb`): Load the raw dataset and get a structural understanding: column types, missing value rates, value ranges, obvious anomalies. The goal is not to find patterns yet, but to catalogue problems.

**Cleaning round 1** (`notebooks/02_cleaning_initial.ipynb`): Fix structural issues identified in round 1: duplicates, broken encodings, non game entries (e.g. dlc, test apps), type conversions, and columns with too many nulls to be useful.

### Round 2 — find the patterns, prepare for modelling

**EDA round 2** (`notebooks/03_eda_final.ipynb`): With a structurally clean dataset, analyse distributions, correlations, and genre/tag breakdowns in depth. Define and justify the target variable here. Identify which transformations the data needs before modelling (log transforms, encoding strategies).

**Cleaning round 2 / final prep** (`notebooks/04_cleaning_final.ipynb`): Apply transformations informed by round 2 EDA. Create the target variable. Apply the temporal train/test split. Export the modelling-ready dataset to `data/processed/`.

---

## Methodology

### Temporal train/test split

Unlike random splits, this project uses a **time based split**: games released before 2020 form the training set; games from 2020–2023 form the held-out test set. This prevents data leakage from future games into the model and reflects real-world deployment conditions where a model trained on historical data must generalise to future releases.

### Feature engineering

Features are constructed exclusively from information available at launch:

- **Temporal:** release year, game age at evaluation date
- **Pricing:** log(price + 1), free-to-play binary flag, price tier (free / budget / mid / premium)
- **Genre structure:** number of genres, primary genre, one hot encoding of top genres
- **Tag structure:** number of tags, frequency encoded tag features
- **Text (planned):** sentiment score from short descriptions via VADER

### Models

Three supervised learning models are trained and compared:

| Model | Purpose |
|---|---|
| Linear Regression | Interpretable baseline; establishes minimum viable performance |
| Random Forest | Captures non-linear relationships; provides native feature importances |
| XGBoost | Expected best performer; compatible with SHAP for post-hoc interpretability |

### Evaluation

Beyond standard metrics (RMSE, MAE, R²), the evaluation section includes:

- **Residual analysis by genre** — does the model systematically over/under predict certain genres?
- **Residual analysis by price band** — does performance degrade for free-to-play or high-price titles?
- **SHAP beeswarm plot** — global feature importance with directional effect
- **SHAP dependence plot** — non-linear relationship between price and predicted success
- **High-error case analysis** — manual inspection of the 20 most mis-predicted games

---

## Repository Structure

```
steam-ml-project/
│
├── data/
│   ├── raw/                        # Original Kaggle dataset (not tracked in git)
│   └── processed/                  # Cleaned, modelling ready data
│
├── notebooks/
│   ├── 01_eda_raw.ipynb            # Round 1 EDA — raw data structure and problems
│   ├── 02_cleaning_initial.ipynb   # Round 1 cleaning — structural fixes
│   ├── 03_eda_final.ipynb          # Round 2 EDA — patterns, target variable definition
│   ├── 04_cleaning_final.ipynb     # Round 2 cleaning — final prep, split, export
│   ├── 05_features.ipynb           # Feature engineering
│   ├── 06_modelling.ipynb          # Model training and comparison
│   └── 07_evaluation.ipynb         # SHAP, residual analysis, error cases
│
├── src/
│   ├── data_cleaning.py            # Reusable cleaning functions (extracted from notebooks)
│   ├── features.py                 # Feature engineering pipeline
│   ├── train.py                    # Model training and cross validation
│   └── evaluate.py                 # Metrics and visualisation utilities
│
├── models/                         # Serialised model artefacts
├── app/
│   └── streamlit_app.py
│
├── mlruns/                         # MLflow tracking (auto-generated)
├── requirements.txt
└── README.md
```

---

## Key Findings (updated as project progresses)

*This section is updated iteratively as each stage completes.*

**Round 1 EDA observations:**
- **Dataset Integrity:** Validated 122,611 unique game records across 39 features, confirming strict data uniqueness with zero duplicate rows or `AppID` overlaps.
- **Schema Correction:** Resolved a persistent column shift misalignment caused by pandas incorrectly defaulting `AppID` as the row index (fixed via `index_col=False`), and identified a potentially merged `DiscountDLC count` header requiring further investigation.
- **Data Quality & Type Casting:** Flagged unusable features (`Achievements` at ~99.97% null) and identified critical dtype mismatches requiring conversion: `Release date` (object → datetime), `Metacritic score` (erroneously bool), `Metacritic url` (erroneously int64), and `Average playtime` (object → numeric).
- **Zero-Inflated Engagement:** Uncovered massive right skew across all player activity metrics; the median for Peak CCU, reviews, and playtime is 0, indicating that mean based metrics will be heavily dominated by a few massive outliers.
- **Market Distribution:** Analyzed the pricing spread ($0 to $999.98) and found a highly skewed $2.24 median, reflecting the platform's heavy dominance of free-to-play and budget titles.

**Round 2 EDA observations (current):**
- *(Distribution findings, genre breakdowns, target variable justification)*

**Modelling findings (planned):**
- *(Best model, SHAP findings, error analysis conclusions)*

---

## Limitations and Future Work

**Limitations**

- **Time-on-market bias:** Lifetime review counts favour older games — titles released years ago have had more time to accumulate reviews, build a player base, and benefit from repeated Steam sales cycles. The temporal train/test split prevents data leakage but does not resolve this; a 2010 game and a 2022 game are not directly comparable on raw review counts alone.
- **Unobserved marketing:** The dataset is restricted to on-platform Steam metrics. Pre-release hype, influencer coverage, ad spend, and community building on platforms like Discord or Reddit are not captured, yet likely drive a meaningful share of a game's commercial outcome.
- **Sales proxy:** Review count serves as a proxy for sales volume since Steam does not publish sales figures. The ratio of purchases to reviews varies across genres, price points, and player demographics, so the proxy is imperfect and inconsistent.
- **Platform scope:** The model is trained exclusively on Steam (PC). Findings may not generalise to console or mobile markets, which have different discovery mechanisms, pricing norms, and player behaviour.

**Future work**

- Genre-stratified models to test whether success drivers differ meaningfully across categories (e.g. action vs. simulation vs. indie).
- Incorporating external signals such as Metacritic score coverage or social media presence where available.

---

## Setup

```bash
git clone https://github.com/haoran-png/Predicting-Video-Game-Commercial-Success-on-Steam.git
cd Predicting-Video-Game-Commercial-Success-on-Steam
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) and place it in `data/raw/`.

---

## Author

**[Haoran Jinfu]**
BSc Mathematics with Data Science | [LinkedIn](https://www.linkedin.com/in/haoranjinfu/)
