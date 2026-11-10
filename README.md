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
           │
           ▼
  ┌─────────────────────────────────────────┐
  │  PHASE 1 — Regression Modeling          │
  │  • 10 chronic disease targets           │
  │  • 25 behavioral/social predictors      │
  │  • Random Forest  (per-disease tuned)   │
  │  • Gradient Boosting (GridSearchCV)     │
  │  • Feature importance + validation      │
  └─────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────┐
  │  PHASE 2 — Chronic Health Risk Index    │
  │  • PyTorch neural network (10→32→16→1)  │
  │  • Composite CHRI score (0–100 scale)   │
  │  • Risk tier classification             │
  │  • Folium interactive U.S. map          │
  └─────────────────────────────────────────┘
```

---

## Results

### Random Forest — Train vs Test R²
![RF R² Scores](reports/images/rf_train_test_r2.png)

### Model Comparison Heatmap — Test R² Across All 10 Diseases
![Heatmap R²](reports/images/heatmap_test_r2.png)

### Random Forest — Train vs Test RMSE
![RF RMSE](reports/images/rf_train_test_rmse.png)

### Correlation Validation — Diabetes Predictors
![Diabetes Correlation](reports/images/corr_diabetes_crudeprev.png)

---

## Code Highlights

### Phase 1 — Per-Disease Tuned Random Forest with GridSearchCV

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [150, 250, 350],
    "max_depth": [8, 12, 16],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 3],
}

rf_best_params = {}
for disease in target_cols:
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid.fit(X_scaled, y[disease])
    rf_best_params[disease] = grid.best_params_
```

### Phase 2 — PyTorch Neural Network for CHRI

```python
import torch
import torch.nn as nn

class CHRI_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = CHRI_NN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
```

### CHRI Score — Scale to 0–100 and Classify Risk Tiers

```python
# Scale NN output to 0–100
chri_nn = model(X_tensor).detach().numpy().flatten()
chri_scaled = (chri_nn - chri_nn.min()) / (chri_nn.max() - chri_nn.min()) * 100

df_clean["CHRI"] = chri_scaled
df_clean["CHRI_Tier"] = pd.cut(
    df_clean["CHRI"],
    bins=[0, 33, 66, 100],
    labels=["low", "medium", "high"]
)
```

### Folium Map — Census Tract Risk Visualization

```python
import folium
from folium.plugins import MarkerCluster

m = folium.Map(location=[39.5, -98.35], zoom_start=4)
cluster = MarkerCluster().add_to(m)

def tier_color(t):
    return "green" if t == "low" else "yellow" if t == "medium" else "red"

for _, row in df_map.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=3,
        color=tier_color(row["CHRI_Tier"]),
        fill=True
    ).add_to(cluster)
```

---

## Key Findings

| Model | Avg Test R² | Notes |
|---|---|---|
| Tuned Random Forest | ~0.90+ | Best interpretability, stable across all 10 diseases |
| Tuned Gradient Boosting | Comparable | Less interpretable, more noise-sensitive |

**Top predictors per disease (Random Forest feature importance):**
- **Diabetes** → Self-care disability, physical inactivity, obesity
- **COPD** → Smoking, physical inactivity, lack of preventive care
- **Hypertension** → Obesity, low checkup rates, socioeconomic stress
- **Stroke / CHD** → BP medication, smoking, aging population indicators

**CHRI:** Successfully compressed 10 disease dimensions into a single 0–100 risk score. Tracts in the **High** tier showed 2–3× higher prevalence of stroke, COPD, and CHD compared to the **Low** tier.

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
Download: [CDC PLACES Data Portal](https://www.cdc.gov/places/index.html)

Place the CSV in the project root:
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
├── requirements.txt
├── data/
│   └── README.md                         # Dataset download instructions
├── reports/
│   └── images/                           # Output charts and visualizations
└── README.md
```

---

## Author

**Venu Bandi** — MS Data Science  
[GitHub](https://github.com/VenuBandi)
