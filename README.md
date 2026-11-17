<div align="center">

# CDC Chronic Health Risk Model

### Can a machine learning model trained on health data from most U.S. states accurately predict disease rates in states it has never seen before?

**For stroke, COPD, and heart disease — yes, with high accuracy.**
**For obesity — no, and understanding why reveals something important about American geography.**

<br>

[![Live Map](https://img.shields.io/badge/🗺️%20Interactive%20Map-Live-brightgreen?style=for-the-badge)](https://venubandi.github.io/cdc-chronic-health-risk/)
[![GitHub stars](https://img.shields.io/github/stars/VenuBandi/cdc-chronic-health-risk?style=for-the-badge&color=yellow)](https://github.com/VenuBandi/cdc-chronic-health-risk/stargazers)

[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.x-blue)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.54-purple)](https://shap.readthedocs.io)
[![CDC PLACES 2024](https://img.shields.io/badge/CDC%20PLACES%202024-83k%20tracts-005EA2)](https://www.cdc.gov/places/index.html)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## What This Project Does

The CDC's PLACES dataset reports health statistics for every census tract in the United States — over 83,000 neighborhoods. This project builds machine learning models that use 27 behavioral and social predictors (smoking rates, food insecurity, healthcare access, physical inactivity, etc.) to predict the prevalence of 10 chronic diseases in each neighborhood.

The critical question isn't whether the models fit well on data they've seen. It's whether they *generalize* — can they predict disease rates accurately in entirely new geographic regions they were never trained on?

To test this, 10 states were completely withheld during training. The models never saw a single data point from Arkansas, California, Georgia, Maine, North Carolina, North Dakota, Nevada, South Carolina, Vermont, or Wyoming. Then their predictions were evaluated against the real numbers from those states.

---

## What We Found

Most diseases are predictable anywhere in the country. **Obesity is not — and the reason tells a story about place.**

<div align="center">

| Disease | Accuracy in Withheld States (R²) | Avg. Prediction Error |
|:--|:--:|:--:|
| COPD | **0.966** | ± 0.37 percentage points |
| Stroke | **0.964** | ± 0.16 percentage points |
| Coronary Heart Disease | **0.959** | ± 0.32 percentage points |
| Cancer | **0.945** | ± 0.48 percentage points |
| Diabetes | **0.934** | ± 0.65 percentage points |
| High Blood Pressure Medication | **0.918** | ± 1.40 percentage points |
| High Blood Pressure | **0.913** | ± 1.72 percentage points |
| Arthritis | **0.880** | ± 1.92 percentage points |
| High Cholesterol | **0.802** | ± 1.56 percentage points |
| **Obesity** | **0.617** | ± 3.58 percentage points |

*R² of 1.0 = perfect prediction. R² of 0.0 = no better than guessing the national average.*
*Results from 500-resample bootstrap on 10 withheld states.*

</div>

---

## Why Obesity Is Different

An R² of 0.617 for Obesity means the model explains only about 62% of the variation in obesity rates across withheld states — a sharp drop from the 0.90+ scores on cardiovascular diseases.

**This is not a modeling failure. It is a geographic finding.**

When we mapped the prediction errors, a clear pattern emerged: the model struggles most in specific regions — rural Appalachia, the Mississippi Delta, parts of the Mountain West. These are places where obesity rates are driven by factors that cannot be measured in a national survey:

- **Food environment** — the density of fast food vs. grocery stores within driving distance
- **Built environment** — whether a neighborhood was designed for walking or only for cars
- **Local food culture** — deeply regional eating patterns that persist across generations

These factors cluster geographically. They are invisible to the 27 behavioral predictors drawn from the BRFSS telephone survey, and that invisibility is precisely what the model's error reveals.

We confirmed this statistically: Moran's I = 0.494 on the obesity prediction errors (z=64.4, p<0.001), meaning the errors are not random noise — they are spatially clustered. The model's failures are neighbors of each other on the map.

Even switching to XGBoost (a more powerful algorithm) only improves obesity accuracy to R²=0.734. The gap isn't a model limitation — it reflects what the available data cannot capture.

---

## What Drives Each Disease (SHAP Analysis)

SHAP values reveal *which predictors* the model relies on most for each disease and *in which direction* — allowing us to check whether the model has learned medically sensible patterns.

**Consistent across all diseases:** Poor physical health days (PHLTH) and poor general health (GHLTH) are the most stable predictors — they are proxies for accumulated disease burden that shows up regardless of which specific condition is being predicted.

**Disease-specific patterns:**

- **Stroke, CHD, COPD** — physical health days and hearing difficulty dominate, reflecting the shared burden of aging-related multimorbidity
- **COPD** — smoking rate (CSMOKING) enters the top 5, as expected for the one disease with the strongest smoking-causation link
- **Obesity** — physical inactivity (LPA) and smoking dominate; both co-cluster geographically in the same rural low-SES communities where obesity is highest, explaining why the model learns the correlation but fails when that geography is absent from training

---

## A Single Risk Score for Every Neighborhood

Diseases do not strike in isolation. The same neighborhoods that have high stroke rates tend to also have high diabetes rates and high COPD rates. A single-disease prediction misses this multimorbidity.

**Phase 2 of this project compresses all 10 disease predictions into one number: the Composite Health Risk Index (CHRI).**

The method is Principal Component Analysis applied to the 10-disease prediction matrix. The first principal component explains **69.5% of all cross-disease variation** — meaning one dimension captures most of what distinguishes a high-burden neighborhood from a low-burden one.

Neighborhoods are then grouped into three tiers:

| Risk Tier | Neighborhoods | What it looks like |
|:--|:--:|:--|
| **Low Risk** | 18,919 | Modest disease burden, strong preventive care engagement |
| **Medium Risk** | 38,292 | Mixed cardiometabolic burden, partial care engagement |
| **High Risk** | 21,234 | Elevated cardiometabolic burden, low preventive care engagement |

These tiers are **95% stable** across 10 different random initializations — they reflect real population groupings, not an artifact of the clustering algorithm.

---

## Where Existing Indices Get It Wrong

The CDC's Social Vulnerability Index (SVI) and the Area Deprivation Index (ADI) are the standard tools for identifying at-risk communities. CHRI is moderately correlated with both (r=0.291 and r=0.631 respectively), but it is not redundant.

Of 76,895 neighborhoods with both CHRI and SVI data, **2,323 (3%) are classified differently** by the two indices:

- **1,563 neighborhoods**: SVI says *high risk*, CHRI says *low risk* — these communities face structural poverty but have younger populations and strong preventive care. They are over-identified as health-burdened by existing tools.
- **760 neighborhoods**: SVI says *low risk*, CHRI says *high risk* — these are socioeconomically secure but carry high disease burden from aging populations (CHD prevalence 10.2%, Cancer 13.5%). They are systematically *missed* by existing tools.

Policy dollars targeted by SVI alone would underfund these 760 high-burden, low-vulnerability communities.

---

## Explore the Map

Every U.S. census tract, colored by composite risk tier.

**[→ Open the interactive map](https://venubandi.github.io/cdc-chronic-health-risk/)**

---

## Technical Details

<details>
<summary><strong>Pipeline architecture</strong></summary>

```
CDC PLACES 2024 (83,522 tracts · 27 predictors · 10 disease targets)
        ↓
  Leakage-free preprocessing
  sklearn.Pipeline([imputer → scaler → model]) fitted on X_train only
  GridSearchCV inside CV folds — test data never touches preprocessing
        ↓
  Three evaluation regimes:
    (1) Random 80/20 split       — baseline (inflated by spatial autocorrelation)
    (2) County-holdout           — 20% of counties withheld
    (3) State-holdout            — 10 full states withheld (reported result)
        ↓
  Algorithm comparison — RF · XGBoost · LightGBM
        ↓
  SHAP (TreeExplainer) on state-holdout test set
        ↓
  CHRI — PCA · k-means (k=3) · external validation
        ↓
  Interactive choropleth (Folium · GitHub Pages)
```

**27 predictors** across 6 groups: Preventive Care (6), Behavioral Risk (4), Health Status (5), Social Determinants of Health (7), Disability (3), Population (2)

</details>

<details>
<summary><strong>Why we use state-holdout instead of random split</strong></summary>

Census tracts in the same county or state share unmeasured characteristics — local policy, regional culture, environmental exposures. A random 80/20 split places neighboring tracts in both train and test, allowing the model to "memorize" local patterns. The resulting R² scores look impressive but overestimate real-world performance.

State-holdout cross-validation removes this contamination by ensuring that every test tract is geographically isolated from every training tract by at least one state boundary. The resulting R² scores are lower — but they represent what the model actually achieves when deployed somewhere it has never seen.

</details>

<details>
<summary><strong>Leakage fix applied in this pipeline</strong></summary>

The original pipeline fit the imputer and scaler on the full dataset before splitting, inflating all metrics. The corrected pipeline wraps everything in `sklearn.Pipeline`:

```python
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=350, max_depth=15, max_features='sqrt')),
])
grid = GridSearchCV(pipe, param_grid, cv=3, scoring='neg_mean_absolute_error')
grid.fit(X_train, y_train)   # test data never enters this call
```

</details>

---

## Quick Start

```bash
git clone https://github.com/VenuBandi/cdc-chronic-health-risk
cd cdc-chronic-health-risk
pip install -r requirements.txt
jupyter lab experiments/corrected_pipeline.ipynb
```

Data: [CDC PLACES 2024](https://www.cdc.gov/places/index.html) — download the census-tract CSV and place it in the repo root.

**Stack:** `Python` · `scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` · `scipy` · `pandas` · `Folium` · `Jupyter`

---

<div align="center">

*Pre-print in preparation · Journal of Biomedical Informatics (target submission)*

If this project was useful, a ⭐ helps others find it.

</div>
