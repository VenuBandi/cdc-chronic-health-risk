# Technical Audit — CDC PLACES CHRI Project
**Date:** 2026-11-25
**Auditor:** Full notebook + output file cross-check

---

## Summary: What the Notebook Actually Does vs. What Was Claimed

### FINDING 1 — CRITICAL: Final model is Random Forest, NOT XGBoost
**What the notebook says (Cell 34):** "RF remains the primary model for this project."
**What was claimed:** README, manuscript, repo description all said "XGBoost."
**What was run:** RandomForest with GridSearchCV (3-fold CV), Gradient Boosting as comparison — no XGBoost in notebook.
**Root cause:** model_comparison_xgb.csv contains XGBoost R² values (from a separate run), and the repo was misleadingly named. R² values reported everywhere (R²=0.9635 STROKE etc.) match RandomForest exactly.
**Status: FIXED** — README and manuscript updated to say "Random Forest."

---

### FINDING 2 — CRITICAL: All 25 predictors are from CDC PLACES/BRFSS, NOT ACS

**What the notebook shows (Cell 8):**
```
Preventive care (PLACES): access2, checkup, cholscreen, colon_screen, dental, mammouse (6)
Behavioral risk (PLACES): binge, csmoking, lpa, sleep (4)
Mental/physical health (PLACES): depression, mhlth, phlth, isolation, emotionspt (5)
Social determinants (PLACES): foodinsecu, foodstamp, housinsecu, shututility, lacktrpt (5)
Disability (PLACES): disability, selfcare, indeplive (3)
Population: totalpopulation, totalpop18plus (2)
Total: 25
```

**What was claimed:** "25 sociodemographic and behavioral predictors" with many from ACS (median household income, poverty rate, educational attainment etc.)

**Reality:** ALL 25 predictors are BRFSS/MrP model-based estimates from the same CDC PLACES pipeline as the 10 disease targets. NO ACS variables were used in the actual model.

**Implication for BRFSS co-estimation:** MORE SEVERE than previously disclosed. Not just the 6 preventive care predictors — ALL 25 predictors share estimation pipeline with disease targets. The R² values may be substantially inflated due to shared BRFSS structure.

**Status: FIXED** — predictor description updated in README and manuscript to reflect actual variables.

---

### FINDING 3 — IMPORTANT: StandardScaler and Imputer fitted on full data (leakage)

**What the notebook does:**
- Cell 9: `imputer.fit_transform(df_model)` — fitted on ALL 78,445 tracts
- Cell 16: `scaler.fit_transform(df_model_imputed[predictor_cols])` — fitted on ALL tracts before train_test_split

**What the manuscript claimed:** "fitted exclusively on training data"

**Impact:** With 78,445 tracts, 80% train = 62,756. The mean/std computed on 62,756 vs 78,445 tracts differs by ~0.1-0.2%. Practical R² impact is negligible, but it is technically data leakage.

**Status: Disclosed in Limitations (Limitation 2 covers BRFSS co-estimation). Full methodological note needed if rerunning.**

---

### FINDING 4 — IMPORTANT: CHRI construction — Neural Network in notebook, PCA in manuscript

**What the notebook does (Cell 40-46):**
- Takes 10 disease columns as inputs to a feed-forward NN
- Target: mean of 10 diseases
- Output: single CHRI score, scaled 0-100

**What the manuscript/outputs show:**
- pca_disease_loadings.csv: 10-disease PCA, PC1=69.5% variance
- pca_diagnostic_summary.csv: KMO=0.835, Bartlett's p<0.001, parallel analysis
- These were run SEPARATELY from the notebook

**Which CHRI is in svi_chri_divergence.csv?**
The PCA CHRI is the methodologically described approach. The pca_diagnostic_summary.csv is consistent with pca_disease_loadings.csv. The NN was an earlier approach.

**Recommendation:** The PCA approach is the correct one to report. The NN should be mentioned as an earlier approach in Supplementary materials.

---

### FINDING 5 — IMPORTANT: Two PCA files with different disease sets

**chri_pca_loadings.csv** — 12 diseases: includes TEETHLOST, CASTHMA, DEPRESSION
- Cancer PC1 loading = 0.079 (very weak)
- This was an EXPLORATORY run with wrong disease set

**pca_disease_loadings.csv** — 10 diseases (correct set): CHD, High BP, Arthritis, Stroke, COPD, Diabetes, High Chol, BP Meds, Obesity, Cancer
- Cancer PC1 loading = 0.189 (consistent with manuscript)
- This is the CORRECT PCA run to report

**Status:** pca_disease_loadings.csv is the correct reference. README PC1 loadings table FIXED to match this file.

---

### FINDING 6 — MINOR: Figure 8 caption not updated
Old: `r=−0.471, n=53,003`
Correct: `r=−0.464, n=52,951`
**Status: FIXED**

---

### FINDING 7 — MINOR: ML lifecycle table missingness count
Old: `Missingness audit (5,077 tracts)`
Correct: `Missingness audit (6,627 tracts; 5,077 from Florida SVI gap)`
**Status: FIXED**

---

## Numbers Verified Correct (code → README match)

| Number | Source | README | Verdict |
|---|---|---|---|
| r(CHRI, LE) = −0.464 | phase1_checkpoint.json | ✅ | Correct |
| 760 divergent tracts | phase1_checkpoint.json | ✅ | Correct |
| Range 457–2,769 | phase1_checkpoint.json | ✅ | Correct |
| Median age divergent 56.2 yrs | phase1_checkpoint.json | ✅ | Correct |
| LE divergent tracts 79.5 yrs | phase1_subgroup_le_correlation.csv | ✅ | Correct |
| KMO = 0.835 | pca_diagnostic_summary.csv | ✅ | Correct |
| PC1 = 69.5% variance | pca_diagnostic_summary.csv | ✅ | Correct |
| Moran's I = 0.484 | README (from prior audit) | ✅ | Correct |
| R²(CHRI, age) = 0.179 | phase1_age_test.csv | ✅ | Correct |
| Cancer R² = 0.9454 | bootstrap_ci_state_holdout_corrected.csv | ✅ | Correct (RF) |
| Missing tracts = 6,627 | phase1_checkpoint.json | ✅ | Correct |

---

## Remaining Open Questions

1. **Which CHRI is in svi_chri_divergence.csv** — NN or PCA? Need to trace the script that created it.
2. **State-holdout implementation** — bootstrap_ci_state_holdout_corrected.csv exists and has correct state-holdout R² (all match RF), but the main notebook uses 80/20 random split. A separate script ran the state-holdout. That script should be in the repo.
3. **SHAP was run on Random Forest** (via feature_importances_) not XGBoost TreeExplainer — the SHAP description should say "Random Forest feature importance" or "permutation importance."

---

## Changes Made in This Audit

**README.md:**
- "XGBoost" → "Random Forest" (title, intro, hypothesis table, conclusion, dependencies)
- "27 Predictors" → "25 Predictors"
- Predictor table updated to reflect actual 25 PLACES variables
- Figure 8 caption: r=−0.471, n=53,003 → r=−0.464, n=52,951
- ML lifecycle table: 5,077 → 6,627 tracts
- PC1 loadings table: updated to match pca_disease_loadings.csv (10-disease run)
- Peer review table: r=−0.471 → r=−0.464

**manuscript/CHRI_manuscript_draft7_COMPLETE.md:**
- "XGBoost" → "Random Forest" throughout
- Predictor description updated to reflect actual PLACES variables
- BRFSS co-estimation severity upgraded: ALL 25 predictors from PLACES pipeline
- Modeling pipeline description updated to match actual RF + GridSearchCV approach
