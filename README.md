# CDC Chronic Health Risk Modeling
### Predicting Community Disease Burden Across 73,000+ U.S. Census Tracts

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Folium](https://img.shields.io/badge/Folium-Maps-77B829)
![Data](https://img.shields.io/badge/Data-CDC%20PLACES%202024-005EA2)

> A two-phase machine learning pipeline that predicts chronic disease prevalence across U.S. census tracts, builds a neural-network-powered Chronic Health Risk Index (CHRI), and maps community-level health burden nationwide.

---

## The Problem

Chronic diseases — diabetes, hypertension, obesity, COPD, stroke — don't strike communities randomly. They cluster geographically, driven by shared behavioral, social, and economic conditions. Yet public health data typically reports each disease in isolation, making it hard to see the full burden on a community.

**This project answers four questions:**
1. How strongly do behavioral and social predictors explain chronic disease rates?
2. Which specific factors drive each disease?
3. Can we build a single composite score that captures total community health risk?
4. Can census tracts be reliably classified into Low / Medium / High risk?

---

## Pipeline Overview

```
CDC PLACES 2024 Dataset (~73,000 census tracts)
           |
           v
  ┌─────────────────────────────────────────┐
  │  PHASE 1 — Regression Modeling          │
  │  • 10 chronic disease targets           │
  │  • 25 behavioral/social predictors      │
  │  • Random Forest  (per-disease tuned)   │
  │  • Gradient Boosting (GridSearchCV)     │
  │  • Feature importance + validation      │
  └─────────────────────────────────────────┘
           |
           v
  ┌─────────────────────────────────────────┐
  │  PHASE 2 — Chronic Health Risk Index    │
  │  • PyTorch neural network (10→32→16→1)  │
  │  • Composite CHRI score (0–100 scale)   │
  │  • Risk tier classification             │
  │  • Folium interactive U.S. map          │
  └─────────────────────────────────────────┘
```

---

## Key Results

| Model | Avg Test R² | Strength |
|---|---|---|
| Tuned Random Forest | ~0.90+ | Best interpretability, stable across all 10 diseases |
| Tuned Gradient Boosting | Comparable | Less interpretable, more noise-sensitive |

**Top predictors identified (across diseases):**
- Self-care disability, smoking, physical inactivity → Diabetes, COPD
- Preventive care access, checkup rates → Hypertension, Cholesterol
- Socioeconomic stress indicators → Stroke, CHD

**CHRI:** Neural network successfully compressed 10 disease dimensions into a single 0–100 risk score. Census tracts classified into Low / Medium / High tiers with high fidelity to underlying disease patterns.

---

## Tech Stack

| Category | Tools |
|---|---|
| Data Processing | Pandas, NumPy, scikit-learn (SimpleImputer, StandardScaler) |
| Machine Learning | Random Forest, Gradient Boosting, GridSearchCV |
| Deep Learning | PyTorch (custom feed-forward NN) |
| Visualization | Matplotlib, Seaborn, Folium + MarkerCluster |
| Data Source | [CDC PLACES 2024](https://www.cdc.gov/places/index.html) — Census Tract level |

---

## Dataset

Data is sourced from the **CDC PLACES 2024 release** (GIS-friendly format, Census Tract level).

Download it here: [CDC PLACES Data Portal](https://www.cdc.gov/places/index.html)

Place the CSV file in the project root before running the notebook:
```
PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2024_release_20251010.csv
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/VenuBandi/cdc-chronic-health-risk.git
cd cdc-chronic-health-risk

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from CDC PLACES portal and place in root directory

# 4. Open and run the notebook
jupyter lab CDC_Chronic_Health_Risk_Model.ipynb
```

---

## Project Structure

```
cdc-chronic-health-risk/
├── CDC_Chronic_Health_Risk_Model.ipynb   # Full two-phase ML pipeline
├── requirements.txt                      # Python dependencies
├── data/
│   └── README.md                         # Data download instructions
├── reports/                              # Output charts and summaries
└── README.md
```

---

## Research Questions Answered

- **RQ1 — Predictive Strength:** Behavioral and social determinants explain >90% of variance in chronic disease rates (R² > 0.90) using tuned Random Forest models.
- **RQ2 — Feature Importance:** Identified top predictors per disease; validated against Pearson correlations to confirm medical alignment.
- **RQ3 — CHRI Construction:** Nonlinear PyTorch NN (10→32→16→1) outperforms linear approaches by capturing co-occurrence and joint risk amplification across diseases.
- **RQ4 — Risk Classification:** CHRI reliably separates census tracts into Low / Medium / High tiers, confirmed by correlation analysis with individual disease outcomes.

---

## Author

**Venu Bandi** — MS Data Science  
[GitHub](https://github.com/VenuBandi)
