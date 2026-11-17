# CDC Chronic Disease Burden Mapping & Prediction

**ML pipeline predicting 10 chronic disease rates across 78,445 US census tracts — state-holdout validated — with Composite Health Risk Index (CHRI)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Venu Bandi** · MPS Data Science, UMBC · [venubandi3060@gmail.com](mailto:venubandi3060@gmail.com)  
> Live map: [venubandi.github.io/cdc-chronic-health-risk](https://venubandi.github.io/cdc-chronic-health-risk)

---

## What This Project Does

This project uses **CDC PLACES 2024** data (78,445 US census tracts) to:

1. **Predict 10 chronic disease rates** using XGBoost/LightGBM models validated on states never seen during training
2. **Build a Composite Health Risk Index (CHRI)** via PCA that ranks every census tract by shared chronic disease burden
3. **Explain disease drivers** using SHAP — identifying which behavioral and social factors drive each disease
4. **Find communities standard tools miss** — 760 tracts with high disease burden not flagged by the Social Vulnerability Index

---

## State-Holdout Prediction Results

All models validated on **10 fully withheld states** — no data from those states was used in training or tuning.

| Disease | R² (State-Holdout) | 95% CI | MAE (pp) |
|---|---|---|---|
| **Cancer** | **0.9454** | [0.9436, 0.9474] | **0.48** |
| COPD | 0.9664 | [0.9655, 0.9674] | 0.37 |
| Stroke | 0.9635 | [0.9604, 0.9661] | 0.16 |
| Coronary Heart Disease | 0.9593 | [0.9552, 0.9619] | 0.32 |
| Diabetes | 0.9342 | [0.9305, 0.9370] | 0.65 |
| BP Medication Use | 0.9176 | [0.9141, 0.9212] | 1.40 |
| High Blood Pressure | 0.9130 | [0.9103, 0.9154] | 1.72 |
| Arthritis | 0.8795 | [0.8766, 0.8825] | 1.92 |
| High Cholesterol | 0.8023 | [0.7953, 0.8089] | 1.56 |
| Obesity | 0.6170 | [0.6057, 0.6266] | 3.58 |

> Cancer predicts prevalence in unseen states to within **0.48 percentage points** on average.

---

## Composite Health Risk Index (CHRI)

CHRI synthesizes all 10 disease measures into a single 0–100 score per census tract using PCA.

- **PC1 explains 69.5%** of total shared variance across all 10 diseases
- All 10 disease loadings are **positive** — CHRI is a direction of maximum shared burden
- Cancer loads primarily on **PC2** (loading 0.610 vs 0.189 on PC1) — it follows a structurally distinct geographic pattern

### PCA Diagnostics

| Test | Result | Verdict |
|---|---|---|
| KMO (overall) | 0.835 | Meritorious — PCA justified |
| Bartlett's Test | χ²=1,300,955, p<0.001 | Data is factorizable |
| Parallel Analysis | Retain PC1 + PC2 | 2 real dimensions |
| Bootstrap CIs (1,000 resamples) | All 10 loadings stable, none cross zero | Stable |
| Leave-one-out sensitivity | 95.2–97.2% top-20% tract agreement | Stable |
| Cancer individual KMO | 0.595 | Structurally distinct from cardiometabolic cluster |

### External Validation — Life Expectancy

CHRI was validated against **USALEEP** tract-level life expectancy (independent dataset, not used in construction):

- **r = −0.471** (p < 0.001) across 53,003 matched tracts
- **4.3-year life expectancy gradient**: High-burden tracts average 75.8 yr vs Low-burden 80.1 yr

### Age Confounding Test

Direct regression of CHRI on **ACS 2022 tract median age** (B01002_001E, 78,426 tracts):

- **R² = 0.179** — only 18% of CHRI variance explained by population age
- CHRI is **NOT primarily an Age Index** (threshold: R² > 0.50)
- High-burden tracts are 6.9 years older on average, but age is not the driver

### Spatial Autocorrelation

- **Moran's I = 0.484** (p < 0.001, 999 permutations) on CHRI scores
- High-burden tracts cluster geographically — standard regression is inappropriate for downstream analyses

---

## 760 Communities Standard Tools Miss

**760 census tracts** fall in the "SVI-low / CHRI-high" quadrant:
- Not flagged as high-risk by the Social Vulnerability Index
- But placed in the **high disease burden tier** by CHRI
- These tracts carry: **Cancer prevalence 13.5%**, **CHD 10.2%** — both above national average
- They are predominantly older, non-impoverished communities in a blind spot of standard resource allocation

---

## SHAP Explainability — Cancer's Distinct Profile

SHAP analysis reveals cancer's geographic distribution is **structurally different** from cardiometabolic diseases:

- Cardiometabolic diseases (Stroke, CHD, COPD, Diabetes): driven by **smoking, physical inactivity, general health status**
- **Cancer**: driven more by **preventive care access** (mammography, colonoscopy rates) and **social isolation** — not metabolic risk factors

This explains why cancer follows a separate geographic axis (PC2) and why standard SES tools fail to identify high-cancer communities.

---

## Dataset

- **Source:** [CDC PLACES 2024](https://www.cdc.gov/places) — Local Data for Better Health, census tract level
- **Estimation method:** BRFSS + Multilevel Regression and Poststratification (MrP)
- **Coverage:** 83,522 census tracts total; 78,445 with complete data across all 10 diseases
- **Excluded:** 5,077 tracts (6.1%) — structural CDC suppression on BPHIGH, HIGHCHOL, BPMED
- **Predictors:** 27 variables — 14 CDC PLACES behavioral/preventive care + 13 ACS socioeconomic
- **Data year:** Approximately 2021–2022 BRFSS

---

## Validation Methodology

> **Why state-holdout instead of random split?**
>
> Random train/test splits on spatial data are optimistic — nearby tracts share characteristics, so the model "sees" the test set's neighborhood during training. State-holdout validation withholds all tracts from 10 entire states, testing how well the model generalizes to **completely new geographic regions**. This is closer to real-world deployment.

---

## Independent Peer Review

Methodology reviewed independently by **7 large language models** acting as statistical reviewers:

- DeepSeek-R1:14b, Qwen3:14b, Phi4:14b, GPT-oss-120B, Qwen3.8-27b, Gemini 3.1 Flash Lite, Cohere Command-A-03-2025
- All 7 consensus concerns identified and resolved (BPMED denominator, age confounding, external validation, Moran's I, circular validation, shared BRFSS inflation, PCA diagnostics)

---

## Repository Structure

```
CDC_Chronic_Health_Risk_Model.ipynb   ← Main notebook (all 10 diseases + CHRI)
experiments/                          ← Corrected pipeline, SHAP analysis, spatial CV
outputs/                              ← Figures, result tables (CSV), ACS age data
docs/                                 ← Methodology documentation
index.html / map.html                 ← Interactive choropleth map (GitHub Pages)
requirements.txt                      ← Python dependencies
```

---

## Setup

```bash
git clone https://github.com/VenuBandi/cdc-chronic-health-risk.git
cd cdc-chronic-health-risk
pip install -r requirements.txt
```

Download the CDC PLACES 2024 census-tract CSV from [data.cdc.gov](https://data.cdc.gov) and place it in the repo root.

**Key dependencies:** scikit-learn, XGBoost, LightGBM, SHAP, pandas, numpy, geopandas, libpysal, matplotlib

---

## Status

Manuscript in preparation. Target journals: *Preventing Chronic Disease* (CDC) or *American Journal of Public Health*.

---

## Author

**Venu Bandi**  
MPS Data Science, University of Maryland Baltimore County (UMBC)  
[venubandi3060@gmail.com](mailto:venubandi3060@gmail.com) · [github.com/VenuBandi](https://github.com/VenuBandi)
