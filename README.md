<div align="center">

# CDC Chronic Health Risk Model

A leakage-free ML pipeline that predicts 10 chronic disease rates across **78,000+ U.S. census tracts**, evaluates geographic generalizability via state-holdout cross-validation, and compresses multi-disease burden into a single PCA-weighted risk index on a live interactive map.

[![Live Map](https://img.shields.io/badge/🗺️%20Interactive%20Map-Live-brightgreen?style=for-the-badge)](https://venubandi.github.io/cdc-chronic-health-risk/)
[![GitHub stars](https://img.shields.io/github/stars/VenuBandi/cdc-chronic-health-risk?style=for-the-badge&color=yellow)](https://github.com/VenuBandi/cdc-chronic-health-risk/stargazers)

[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.x-blue)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.54-purple)](https://shap.readthedocs.io)
[![CDC PLACES 2024](https://img.shields.io/badge/CDC%20PLACES%202024-78k%20tracts-005EA2)](https://www.cdc.gov/places/index.html)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

<br>

---

## The Central Finding

> **Cardiovascular and pulmonary outcomes generalize strongly across U.S. geographies. Obesity does not.**

Under state-holdout cross-validation (10 withheld states), Random Forest models achieve R²=0.964 for Stroke and R²=0.966 for COPD. Obesity drops to R²=0.617 — a gap that persists even after expanding the predictor set to include transportation access and social support (27 predictors total).

Moran's I = 0.494 (z=64.4, p<0.001) on Obesity residuals confirms the gap is driven by spatial autocorrelation: unmeasured food and built environment factors cluster geographically in ways BRFSS-based predictors cannot capture. Even XGBoost only recovers Obesity to R²=0.734 — the gap persists across model families.

---

## State-Holdout Results (10 withheld states)

Three evaluation regimes were run. Only the state-holdout results are reported here — random-split R² is inflated by spatial autocorrelation between adjacent tracts and is provided in the paper for comparison only.

| Disease | State-HO MAE (pp) | 95% CI | State-HO R² | 95% CI |
|:--|:--:|:--:|:--:|:--:|
| COPD | 0.370 | [0.365–0.374] | **0.966** | [0.966–0.967] |
| Stroke | 0.159 | [0.157–0.162] | **0.964** | [0.960–0.966] |
| CHD | 0.317 | [0.313–0.321] | **0.959** | [0.955–0.962] |
| Cancer | 0.478 | [0.472–0.483] | **0.945** | [0.944–0.947] |
| Diabetes | 0.650 | [0.641–0.659] | **0.934** | [0.931–0.937] |
| BPMED | 1.398 | [1.379–1.416] | **0.918** | [0.914–0.921] |
| BPHIGH | 1.723 | [1.703–1.741] | **0.913** | [0.910–0.915] |
| Arthritis | 1.917 | [1.897–1.936] | **0.880** | [0.877–0.883] |
| High Cholesterol | 1.561 | [1.544–1.579] | **0.802** | [0.795–0.809] |
| **Obesity** | **3.579** | [3.541–3.621] | **0.617** | [0.606–0.627] |

*95% CIs from 500 bootstrap resamples on state-holdout test set. Withheld states: AR, CA, GA, ME, NC, ND, NV, SC, VT, WY.*

---

## What the Model Found

### 1 · Disease mechanism drives generalizability
Cardiovascular and pulmonary outcomes (Stroke, COPD, CHD) generalize because their BRFSS predictors — physical health days, smoking, preventive care — are nationally consistent risk factors. Obesity generalizes poorly because its residuals cluster spatially: California high-error tracts have MAE=4.45 pp vs Wyoming's 1.62 pp. The unmeasured confounders (food deserts, walkability, local food culture) are geographically concentrated and invisible to the predictor set.

### 2 · Detection effects for screened conditions
For asymptomatic diseases (Hypertension, High Cholesterol, Cancer), **screening and checkup rates** are the top predictors — not lifestyle factors. Communities with low healthcare access have *undiagnosed* disease, not lower disease prevalence.

### 3 · SHAP stability reveals model confidence
Physical health status (Phlth, stability=0.585) and general health status (Ghlth, 0.581) are the most stable national predictors — their SHAP contributions are consistent across all 10 diseases. Smoking (Csmoking, stability=0.126) is the least stable: near-zero for Stroke, highest for Obesity — an epidemiologically interpretable pattern where smoking and obesity co-cluster in rural communities.

---

## Phase 2 — Composite Health Risk Index (CHRI)

Chronic diseases co-occur: the same census tracts carry elevated stroke *and* COPD *and* diabetes. A single-disease model misses multimorbidity.

**Method:** Principal Component Analysis on the 10-disease prediction matrix (78,445 tracts). PC1 explains **69.5% of cross-disease variance** with all 10 diseases loading positively (highest: CHD 0.363, BPHIGH 0.361, Arthritis 0.351).

**Validation:**
- Pearson r = 0.994 vs arithmetic mean CHRI — PCA weighting is scientifically defensible but not arbitrary
- k-means k=3 tiers (Low / Medium / High) are **95% stable** across 10 random seeds — the tiers are real population groupings
- CHRI vs SVI (CDC Social Vulnerability Index): r=0.291 — CHRI captures disease burden not duplicated by existing social vulnerability indices
- CHRI vs ADI (Area Deprivation Index): r=0.631 — moderate overlap with deprivation, but CHRI adds independent disease-burden signal

**Risk tier characteristics:**

| Tier | Tracts | Profile |
|:--|:--:|:--|
| **Low** | 18,919 | Modest disease burden, strong preventive care access |
| **Medium** | 38,292 | Mixed cardiometabolic burden, partial care engagement |
| **High** | 21,234 | Elevated cardiometabolic burden, low preventive-care engagement |

---

## Divergence Analysis — Where SVI and CHRI Disagree

Of 76,895 tracts with both CHRI and SVI data:
- **1,563 tracts (2.0%)**: SVI high / CHRI low — structurally vulnerable but relatively low disease burden (young demographics, strong preventive care despite poverty)
- **760 tracts (1.0%)**: SVI low / CHRI high — socioeconomically secure but high disease burden (aging populations with accumulated conditions; highest CHD 10.2%, Cancer 13.5%)

These are the tracts existing indices misclassify. Policy interventions targeting SVI alone would systematically underfund the 760 high-burden, low-vulnerability tracts.

---

## Interactive Map

Every U.S. census tract colored by composite risk tier.

**[→ Open the live map](https://venubandi.github.io/cdc-chronic-health-risk/)**

---

## Pipeline

```
CDC PLACES 2024 (78k tracts · 27 predictors · 10 disease targets)
        ↓
  Leakage-free preprocessing — Pipeline([imputer, scaler, model])
  fitted on X_train only; GridSearchCV inside CV folds
        ↓
  Three evaluation regimes:
    (1) Random 80/20 split — baseline (inflated; spatial autocorrelation)
    (2) County-holdout — 20% counties withheld
    (3) State-holdout — 10 states withheld (publishable result)
        ↓
  Algorithm comparison — RF vs XGBoost vs LightGBM
  XGBoost best on 7/10 diseases; Obesity gap persists (R² 0.617→0.734)
        ↓
  SHAP (TreeExplainer) — 4 diseases × 400 tracts; stability heatmap; Moran's I
        ↓
  CHRI Phase 2 — PCA on 10-disease matrix; PC1=69.5%; k-means tiers; 95% stable
        ↓
  External validation — CHRI vs SVI (r=0.291) · CHRI vs ADI (r=0.631)
  Divergence analysis: 2,323 tracts with SVI/CHRI mismatch
        ↓
  Interactive choropleth map (Folium · GitHub Pages)
```

---

## Leakage Fix

The original v1 pipeline fit the imputer and scaler on the full dataset before splitting, inflating all metrics. The corrected pipeline wraps everything in `sklearn.Pipeline` and fits exclusively on `X_train`:

```python
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=350, max_depth=15, max_features='sqrt')),
])
grid = GridSearchCV(pipe, param_grid, cv=3, scoring='neg_mean_absolute_error')
grid.fit(X_train, y_train)   # never sees test data
```

---

## Quick Start

```bash
git clone https://github.com/VenuBandi/cdc-chronic-health-risk
cd cdc-chronic-health-risk
pip install -r requirements.txt
jupyter lab experiments/corrected_pipeline.ipynb
```

Data: [CDC PLACES 2024](https://www.cdc.gov/places/index.html) — download the census-tract CSV and place it in the repo root.

---

## Tech Stack

`Python` `scikit-learn` `XGBoost` `LightGBM` `SHAP` `scipy` `pandas` `Folium` `Jupyter`

---

<div align="center">

Pre-print in preparation · Journal of Biomedical Informatics (target submission)

If this project was useful, a ⭐ helps others find it.

</div>
