<div align="center">

# CDC Chronic Health Risk Model

A two-phase ML pipeline predicting chronic disease rates across **73,000+ U.S. census tracts**.<br>
Random Forest achieves **R² > 0.90** on 10 disease outcomes. A PyTorch neural network compresses these into a single geographic risk index.

[![Live Map](https://img.shields.io/badge/🗺️%20Interactive%20Map-Live-brightgreen?style=for-the-badge)](https://venubandi.github.io/cdc-chronic-health-risk/)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![CDC PLACES](https://img.shields.io/badge/CDC%20PLACES%202024-73k%20tracts-005EA2)](https://www.cdc.gov/places/index.html)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

<br>

![Model R² Heatmap](reports/images/heatmap_test_r2.png)

> [!NOTE]
> Census tracts in the **High** risk tier show **2–3× higher prevalence** of stroke, COPD, and CHD compared to the Low tier — confirming that the CHRI captures meaningful geographic health stratification beyond any single disease metric.

## Results

| Model | Avg Test R² | Notes |
|:--|:--:|:--|
| Tuned Random Forest | ~0.90+ | Strongest interpretability across all 10 disease targets |
| Tuned Gradient Boosting | Comparable | Less interpretable, more noise-sensitive |

Top predictors by disease (Random Forest feature importance):

| Disease | Key Predictors |
|:--|:--|
| Diabetes | Self-care disability, physical inactivity, obesity |
| COPD | Smoking, physical inactivity, lack of preventive care |
| Hypertension | Obesity, low checkup rates, socioeconomic stress indicators |
| Stroke / CHD | BP medication adherence, smoking, aging population |

<table>
<tr>
<td><img src="reports/images/rf_train_test_r2.png"></td>
<td><img src="reports/images/rf_train_test_rmse.png"></td>
</tr>
<tr><td align="center">Random Forest R²</td><td align="center">Random Forest RMSE</td></tr>
</table>

## Pipeline

**Phase 1 — Regression** · 10 chronic disease targets · 25 behavioral and social predictors · per-disease Random Forest and Gradient Boosting with 5-fold GridSearchCV

**Phase 2 — CHRI** · PyTorch feed-forward network (10→32→16→1) · composite risk score normalized to 0–100 · tier classification (Low / Medium / High) · Folium census tract map

## Quick Start

```bash
git clone https://github.com/VenuBandi/cdc-chronic-health-risk
cd cdc-chronic-health-risk
pip install -r requirements.txt
jupyter lab CDC_Chronic_Health_Risk_Model.ipynb
```

Data: [CDC PLACES 2024](https://www.cdc.gov/places/index.html) → place the CSV in the repo root before running.

## License

MIT
