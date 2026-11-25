# Composite Health Risk Index (CHRI): Identifying US Census Tracts with Elevated Chronic Disease Burden Not Flagged by the Social Vulnerability Index
**Draft 7 — 2026-11-25**
**Author: Venu Bandi**
**Status: Complete draft — all reviewer recommendations implemented**

---

## Abstract

**Background.** Public health programs routinely use the Social Vulnerability Index (SVI) to identify communities for chronic disease resource allocation. Because SVI measures socioeconomic conditions rather than actual disease burden, communities may accumulate substantial chronic disease burden before SVI signals elevated risk — particularly aging communities with favorable socioeconomic profiles but high prevalence of cardiometabolic conditions.

**Objective.** To identify US census tracts carrying elevated chronic disease burden not flagged by SVI, characterize their demographic profile, and assess whether elevated burden precedes detectable mortality-stage outcomes in these communities.

**Methods.** Using CDC PLACES 2024 small-area prevalence estimates for 10 chronic diseases across 78,445 US census tracts, we trained state-holdout cross-validated Random Forest models predicting each disease rate from 25 behavioral, preventive care, and social determinant predictors drawn from CDC PLACES. We derived a Composite Health Risk Index (CHRI) using principal component analysis (PCA) on the 10-disease matrix and validated it externally against census-tract life expectancy data (USALEEP). SHAP explainability analysis identified geographic correlates of modeled disease rates by disease domain.

**Results.** CHRI was inversely correlated with life expectancy across all tracts (r = −0.464, p < 0.001), with a 4.3-year gradient across burden tiers. Between 760 and 1,907 census tracts — depending on threshold definition — were classified as low-SVI but high-CHRI. These communities had a median tract age of 56.2 years (16.8 years above the national median) and near-average life expectancy (79.5 years), indicating elevated disease burden has not yet translated into excess mortality. Cancer geography was structurally distinct from cardiometabolic burden, predicting along a separate geographic dimension associated with preventive care screening utilization rates.

**Conclusions.** CHRI identifies aging communities with accumulated chronic disease burden that SVI does not flag and in which mortality has not yet peaked — the intervention window where preventive allocation is most effective. These findings support using disease-burden indices alongside socioeconomic vulnerability measures in chronic disease resource allocation.

---

## 1. Introduction

Chronic diseases — including coronary heart disease, stroke, diabetes, COPD, and cancer — account for seven in ten deaths in the United States and consume more than 75% of the nation's $4.1 trillion annual healthcare expenditure [CITE: CDC chronic disease overview]. Geographic targeting of chronic disease prevention resources relies heavily on the Social Vulnerability Index (SVI), a CDC composite measure of 15 socioeconomic and housing variables originally developed for disaster preparedness and emergency response [CITE: Flanagan 2011]. Because SVI quantifies socioeconomic deprivation rather than actual disease burden, it systematically overlooks communities where chronic disease prevalence is elevated despite favorable socioeconomic profiles — particularly aging suburban and rural communities where cardiometabolic conditions accumulate in the absence of acute poverty. Prior work has demonstrated that socioeconomic deprivation indices and direct chronic disease burden measures diverge substantially at the census-tract level [CITE: prior SVI-burden divergence paper], yet no published tool directly quantifies this gap at national scale using tract-level disease data.

Accurate census-tract-level prediction of chronic disease rates is the methodological foundation required to build such a tool. Prior studies have predicted individual disease rates at the county or city level using machine learning [CITE: Jiao & Feng 2021], but no study had validated tract-level predictions across all 10 CDC PLACES chronic disease outcomes simultaneously at the national scale, nor tested whether geographic patterns learned in one set of states transfer to states entirely withheld from training. In Phase 1 of this study, we address this gap by training state-holdout cross-validated Random Forest models predicting each of 10 chronic disease rates from 25 behavioral, preventive care, and social determinant predictors across 78,445 US census tracts. We apply SHAP explainability analysis to characterize the geographic correlates of predicted disease rates by disease domain — distinguishing the community-level factors associated with cardiometabolic burden from those associated with cancer geography, which follows a structurally distinct geographic pattern.

In Phase 2, we use the 10 validated disease predictions to construct a Composite Health Risk Index (CHRI) — a single census-tract-level score that ranks communities by shared chronic disease burden, independent of socioeconomic proxies. CHRI is derived via principal component analysis (PCA) on the 10-disease matrix, externally validated against independent census-tract life expectancy data, and directly compared to the SVI to identify divergent communities. We find that between 760 and 1,907 census tracts — depending on threshold definition — carry elevated CHRI scores despite low SVI rankings. These communities are characterized by above-average aging (median tract age 56.2 years, 16.8 years above the national median) and near-average life expectancy (79.5 years), indicating that accumulated chronic disease burden has not yet translated into excess mortality. This pre-mortality-peak profile represents the precise intervention window in which preventive resource allocation is most effective — communities that standard SES-based tools cannot identify and that CHRI is designed to surface.

---

## 2. Methods

### 2.1 Data Sources

**CDC PLACES 2024.** The primary data source is the CDC PLACES 2024 Small Area Health Estimates dataset, which provides model-based chronic disease prevalence estimates for 83,522 US census tracts. Estimates are derived from the 2021–2022 Behavioral Risk Factor Surveillance System (BRFSS) using multilevel regression and poststratification (MrP), with American Community Survey (ACS) 5-year estimates used as small-area covariates. We retained 78,445 tracts with complete data across all 10 disease outcomes; 6,627 tracts were excluded due to missing values, with 5,077 of these attributable to a Florida-specific SVI data gap rather than disease data missingness. Excluded tracts showed similar disease burden profiles to included tracts, with the exception of a 39% higher uninsured rate.

**American Community Survey.** Sociodemographic predictors were drawn from the 2017–2021 ACS 5-year estimates, accessed via the Census API. These include median household income, poverty rate, educational attainment, unemployment rate, insurance coverage, housing characteristics, age composition, racial and ethnic composition, and urban/rural classification.

**USALEEP Life Expectancy.** External validation used tract-level life expectancy at birth from the US Small-Area Life Expectancy Estimates Project (USALEEP), 2010–2015 release — the only publicly available life expectancy dataset at census-tract resolution. A total of 52,951 tracts were matched between CHRI and USALEEP on 11-digit FIPS code.

**Social Vulnerability Index.** SVI overall percentile rankings (RPL_THEMES) were obtained from the CDC 2020 SVI dataset at the census-tract level and used exclusively for the Phase 2 divergence analysis.

**[C3] Data Source Independence — BRFSS Co-Estimation Disclosure.**
The 25 predictors used in Phase 1 derive from two structurally distinct sources. Sociodemographic predictors — including population density, age composition, median household income, educational attainment, insurance coverage, and housing characteristics — are drawn from the ACS 5-year estimates and are estimated independently of the CDC PLACES pipeline. Preventive care utilization predictors — including mammography screening rates, colorectal cancer screening rates, and dental visit rates — are themselves model-based estimates generated from the same BRFSS survey data via the same MrP pipeline used to produce the 10 disease outcome measures. Because BRFSS-derived predictors and disease targets share both the underlying survey respondents and the geographic smoothing model, their spatial correlation patterns are partly induced by shared measurement structure rather than purely by epidemiological association. Models incorporating these BRFSS-derived predictors may therefore exhibit inflated apparent predictive performance. This shared estimation structure is a known property of PLACES-based analyses and does not invalidate the geographic associations reported; it is disclosed here and in Limitations.

### 2.2 Phase 1 — Geographic Disease Prediction

**Predictors.** Twenty-five predictors were drawn entirely from CDC PLACES 2024 model-based estimates across six domains: preventive care utilization (uninsured rate, annual checkup, cholesterol screening, colonoscopy, dental visit, mammography; n=6); behavioral risk factors (binge drinking, smoking, physical inactivity, sleep deprivation; n=4); mental and physical health status (depression, frequent mental distress, frequent physical distress, social isolation, lack of emotional support; n=5); social determinants of health (food insecurity, SNAP use, housing insecurity, utility shutoff, lack of transportation; n=5); disability and functional limitations (disability prevalence, self-care difficulty, independent living difficulty; n=3); and population size (total population, adult population 18+; n=2). All 25 predictors are BRFSS/MrP model-based estimates generated through the same CDC PLACES pipeline as the 10 disease targets; the shared estimation structure is disclosed in the BRFSS Co-Estimation section and in Limitations.

**Target Variables.** Ten chronic disease crude prevalence rates from CDC PLACES 2024 were modeled as separate regression targets: Stroke, Coronary Heart Disease (CHD), COPD, Cancer (all invasive types excluding non-melanoma skin), Diabetes, Arthritis, Obesity, High Blood Pressure (BPHIGH), High Cholesterol (HIGHCHOL), and BP Medication Use (BPMED).

**Modeling Pipeline.** A Random Forest Regressor was trained for each of the 10 disease targets using GridSearchCV (3-fold CV, R² scoring) to select hyperparameters (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features). StandardScaler normalization was applied to all predictors prior to model fitting. Random Forest was selected as the primary model over Gradient Boosting due to superior interpretability, stable feature importance rankings, and alignment with epidemiological literature (see Supplementary Note).

**State-Holdout Validation.** For each disease, 10 states were randomly selected (random seed = 42) and fully withheld from model training and hyperparameter selection. Final models were evaluated on these 10 withheld states, which the model had never seen during development. Bootstrap confidence intervals (200 resamples) were computed on state-holdout MAE and R².

**SHAP Explainability.** SHAP TreeExplainer values were computed on the held-out test set for each disease model to identify the community-level features most strongly associated with predicted disease rates at the census-tract level.

**[C2] SHAP Interpretive Caveat.** SHAP values quantify the marginal contribution of each predictor to the XGBoost model's output at the census-tract level and should be interpreted as geographic correlates of predicted disease rates, not as causal mechanisms. Because all BRFSS-derived predictors and disease targets share the same underlying estimation pipeline (see Section 2.1), SHAP attributions for behavioral predictors such as mammography and colonoscopy rates reflect the statistical structure of the CDC PLACES modeling framework in addition to any true epidemiological association; these attributions are interpreted with corresponding caution throughout.

### 2.3 Phase 2 — Composite Health Risk Index (CHRI)

**PCA Construction.** CHRI was constructed by applying principal component analysis (PCA) to the standardized 10-disease matrix across all 78,445 tracts. All 10 disease variables were standardized to mean=0, SD=1 prior to PCA. The first principal component (PC1) was used as the CHRI score, as it captures the direction of maximum shared variance across diseases. PC1 scores were normalized to a 0–100 scale.

**PCA Diagnostics.** Four diagnostics were run: Kaiser-Meyer-Olkin (KMO) sampling adequacy test (threshold ≥0.80), Bartlett's test of sphericity (threshold p<0.05), parallel analysis to determine components to retain, and bootstrap confidence intervals (1,000 resamples) on PC1 loadings.

**CHRI Tiering.** Census tracts were assigned to Low, Medium, and High burden tiers using k-means clustering (k=3) on CHRI scores. Tier stability was assessed across 10 random seeds.

**External Validation.** CHRI was validated against USALEEP tract-level life expectancy using Pearson correlation on 52,951 matched tracts.

**SVI Divergence Analysis.** Tracts in the bottom 20th percentile of SVI (SVI-low) and top 20th percentile of CHRI (CHRI-high) were classified as divergent. A sensitivity analysis varied both thresholds across a 3×3 grid (SVI ≤ 20th/25th/33rd × CHRI ≥ 75th/80th/85th percentile).

---

## 3. Results

### 3.1 Phase 1 — Geographic Disease Prediction

**Model Performance.** State-holdout cross-validated Random Forest models achieved strong predictive performance across all 10 disease outcomes.

| Disease | R² (State-Holdout) | 95% CI | MAE (pp) |
|---|---|---|---|
| COPD | 0.9664 | [0.9655, 0.9674] | 0.37 |
| Stroke | 0.9635 | [0.9604, 0.9661] | 0.16 |
| Coronary Heart Disease | 0.9593 | [0.9552, 0.9619] | 0.32 |
| Cancer | 0.9454 | [0.9436, 0.9474] | 0.48 |
| Diabetes | 0.9342 | [0.9305, 0.9370] | 0.65 |
| BP Medication Use | 0.9176 | [0.9141, 0.9212] | 1.40 |
| High Blood Pressure | 0.9130 | [0.9103, 0.9154] | 1.72 |
| Arthritis | 0.8795 | [0.8766, 0.8825] | 1.92 |
| High Cholesterol | 0.8023 | [0.7953, 0.8089] | 1.56 |
| Obesity | 0.6170 | [0.6057, 0.6266] | 3.58 |

Cardiometabolic diseases achieved R²≥0.934, indicating strong geographic transferability. Obesity was the most difficult to predict (R²=0.617), consistent with its dependence on local food and built-environment factors.

**[C2] SHAP Explainability — Disease Domain Patterns.** SHAP analysis identified distinct geographic predictor profiles for cardiometabolic diseases versus cancer. All SHAP attributions are on StandardScaler space and represent geographic correlates of predicted disease rates; they should not be interpreted as causal mechanisms.

*Cardiometabolic diseases* (Stroke, CHD, COPD, Diabetes): Frequent physical health days (PHLTH), smoking prevalence (CSMOKING), and general health status (GHLTH) were the geographic features most strongly associated with modeled rates across this disease cluster.

*Cancer*: Cancer rate variation was associated with a structurally different set of geographic predictors. Preventive care screening utilization rates — specifically mammography and colorectal cancer screening — showed the strongest positive associations with modeled cancer rates in our models. Smoking and physical inactivity, the dominant predictors for cardiometabolic diseases, showed weaker associations with predicted cancer rates. This distinct predictor profile is consistent with the structural separation of cancer geography on PC2 (see Section 3.2).

### 3.2 Phase 2 — Composite Health Risk Index (CHRI)

**PCA Diagnostics.**

| Diagnostic | Result | Threshold | Verdict |
|---|---|---|---|
| KMO Overall | 0.835 | ≥0.80 | Meritorious |
| Bartlett's Test | χ²=1,300,955, p<0.001 | p<0.05 | Factorizable |
| Parallel Analysis | Retain PC1 + PC2 | Eigenvalue > random 95th pct | 2 real dimensions |
| Bootstrap CIs on PC1 loadings | All 10 stable | No CI crosses zero | Stable structure |
| Cancer individual KMO | 0.595 | ≥0.60 | Below threshold — structurally distinct |
| BPMED sensitivity | r=0.9973 vs original | r>0.99 | Denominator issue resolved |

PC1 explained 69.5% of total shared variance. Cancer had the lowest PC1 loading (0.189) and highest PC2 loading (0.610), establishing that cancer geography is a structurally real second dimension (PC2 eigenvalue=1.82, above random threshold).

**[C5] Cancer Aggregation Note.** CANCER_CrudePrev aggregates over 100 invasive cancer types with distinct etiologies and geographic distributions. The association between preventive screening utilization and modeled cancer rates in this analysis likely reflects the contribution of screening-detectable cancers (colorectal, breast, cervical). Cancer-type-specific analysis using Cancer InFocus is recommended as a direct extension of this work.

**External Validation.** CHRI was inversely correlated with USALEEP tract-level life expectancy (r=−0.464, p<0.001, n=52,951 matched tracts). The 4.3-year life expectancy gradient across CHRI tiers (Low=80.1 years, High=75.8 years) supports the clinical relevance of tier assignment.

**Age Confounding Test.** R²(CHRI, median age)=0.179. Age composition explains 17.9% of CHRI variance — a real but non-dominant contribution. By a pre-specified bright-line criterion (R²>0.50 = Age Index), CHRI is not primarily an age geography index.

**Spatial Autocorrelation.** Moran's I on CHRI scores = 0.484 (p<0.001, 999 permutations).

**[C6] SVI Divergence — Communities Missed by Standard Allocation.**

| SVI threshold | CHRI ≥ 75th pct | CHRI ≥ 80th pct | CHRI ≥ 85th pct |
|---|---|---|---|
| SVI ≤ 20th pct | 1,117 tracts (~4.5M) | **760 tracts (~3.0M)** | 457 tracts (~1.8M) |
| SVI ≤ 25th pct | 1,662 tracts (~6.6M) | 1,115 tracts (~4.5M) | 665 tracts (~2.7M) |
| SVI ≤ 33rd pct | 2,769 tracts (~11.1M) | 1,907 tracts (~7.6M) | 1,139 tracts (~4.6M) |

At the primary threshold (SVI ≤ 20th percentile, CHRI ≥ 80th percentile), 760 divergent tracts were identified:
- **Median tract age: 56.2 years** (16.8 years above the national median of 39.4)
- **Life expectancy: 79.5 years** (near-national-average — not elevated mortality)
- **Cancer prevalence: 13.5%** (69% above national mean of ~8.0%)
- **CHD prevalence: 10.2%** (above national average)

The near-average life expectancy in these high-CHRI/low-SVI tracts — compared to 73.0 years in tracts classified as high by both indices — indicates that accumulated chronic disease burden has not yet translated into excess mortality. This pre-mortality-peak profile represents the intervention window in which preventive resource allocation is most actionable.

---

## 4. Discussion

### 4.1 Summary of Findings

This study presents a two-phase framework for identifying census-tract communities that carry elevated chronic disease burden despite favorable socioeconomic profiles. In Phase 1, state-holdout cross-validated XGBoost models predicted all 10 CDC PLACES chronic disease rates with high geographic transferability (R²=0.934–0.966 for cardiometabolic diseases). In Phase 2, PCA on the 10-disease matrix produced CHRI — externally validated against tract-level life expectancy (r=−0.464) — identifying between 760 and 1,907 census tracts that carry high disease burden despite low SVI rankings. These communities are predominantly aging (median age 56.2 years) with near-average life expectancy (79.5 years), indicating accumulated disease burden has not yet produced excess mortality.

### 4.2 The Pre-Mortality-Peak Intervention Window

The defining characteristic of divergent tracts — elevated CHRI combined with near-average life expectancy — has direct implications for how chronic disease programs should be timed. Among tracts that rank high on both CHRI and SVI, mean life expectancy is 73.0 years; in divergent tracts, it is 79.5 years. This 6.5-year gap does not reflect lower disease burden in divergent tracts — CHRI is comparably elevated. It reflects that the aging communities in divergent tracts have not yet accumulated the mortality consequences their disease profiles predict.

This pre-mortality-peak window is precisely the period in which preventive interventions have maximum leverage. Once life expectancy in a community drops to 73 years, cardiovascular events, COPD exacerbations, and diabetes complications have already reduced the population susceptible to primary and secondary prevention. At 79.5 years — where divergent tracts currently sit — cardiometabolic management, cancer screening uptake, and behavioral modification can still alter the trajectory. Conventional SES-based targeting misses this window because poverty, the primary SVI signal, is not elevated in these communities. CHRI identifies the window directly.

### 4.3 [C4] Cancer Geography: Screening Utilization as a Correlate, Not a Cause

SHAP analysis identified mammography and colorectal cancer screening utilization rates as among the top geographic correlates of modeled cancer rates — a finding that requires careful interpretation. The positive association between screening utilization and modeled cancer prevalence does not indicate that screening increases cancer rates. Screening-detectable cancers contribute disproportionately to the CANCER_CrudePrev variable, and geographic areas with higher screening uptake detect a higher proportion of prevalent cases that would otherwise go undiagnosed. This detection paradox is well-established in cancer epidemiology: screening programs in a population initially increase apparent prevalence as subclinical disease is surfaced.

The mammography rate and colonoscopy rate variables in CDC PLACES are model-based estimates from the same BRFSS/MrP pipeline that generated the disease targets. Their correlation with cancer rates therefore partly reflects shared estimation structure rather than a purely epidemiological relationship. The SHAP attribution to these predictors should be understood as a geographic structural signal — areas that invest in screening infrastructure also tend to have the sociodemographic characteristics correlated with elevated cancer detection — not as evidence of a causal mechanism.

The aggregate cancer variable further limits interpretive specificity. Cancer-type-specific analysis using Cancer InFocus county-level data would substantially improve mechanistic interpretability and is recommended as the primary extension of this work.

### 4.4 [C1] What XGBoost Models Add Beyond Existing Indices

**At the prediction level**, the value of Phase 1 is that state-holdout validation on 10 entirely withheld states provides evidence that the learned geographic disease patterns transfer across administrative boundaries. A model evaluated on 10 never-seen states offers substantially more generalizability evidence than random 80/20 holdout on spatially autocorrelated data. The R²=0.934–0.966 on withheld states for cardiometabolic diseases indicates that the sociodemographic and behavioral predictors available nationally encode enough of the geographic disease structure to support generalizable prediction — a result that cannot be assumed and needed to be demonstrated.

**At the index construction level**, CHRI is constructed from direct disease burden estimates, not from socioeconomic proxies. SVI does not contain disease data; ADI does not contain disease data; SDOH composite indices do not contain disease data. CHRI is the first national census-tract index that derives its scores from co-occurring chronic disease prevalence rates across 10 conditions simultaneously, validated against independent life expectancy data. The divergent-tract finding cannot be produced by any index built exclusively from socioeconomic covariates, by definition. The incremental contribution of the ML step is specifically to enable this comparison: without validated tract-level disease predictions, a 10-disease PCA at national resolution is not possible from administrative data sources.

### 4.5 Spatial Structure and Policy Implications

Moran's I=0.484 on CHRI scores confirms that high-burden tracts cluster geographically. This has two practical implications. First, public health programs targeting individual high-CHRI tracts will typically find geographic concentrations where intervention infrastructure can serve multiple adjacent tracts simultaneously. Second, the spatial autocorrelation confirms that regression-based studies using CHRI as an outcome variable should apply spatial lag or spatial error correction.

---

## 5. Limitations

1. **Age standardization.** CHRI uses crude prevalence rates; R²(CHRI, age)=0.179 indicates real but non-dominant age confounding. The divergent tract subgroup (median age 56.2) is substantially older than average; age confounding is a significant contributor to this specific finding.

2. **BRFSS co-estimation.** Several Phase 1 predictors (mammography, colonoscopy, dental visit rates) are generated from the same BRFSS pipeline as the disease targets, likely inflating apparent R².

3. **SVI threshold sensitivity.** The divergent tract count varies from 457 to 2,769 across the 3×3 threshold grid. The 760-tract headline is a point estimate; policy applications should report the 760–1,907 range.

4. **USALEEP temporal mismatch.** Life expectancy data (2010–2015) predates the PLACES disease data (2021–2022) by 7–9 years.

5. **Missing tracts — Florida SVI data gap.** 6,627 tracts excluded; 5,077 from Florida-specific SVI gap. Count is a national estimate excluding Florida.

6. **Cancer heterogeneity.** CANCER_CrudePrev aggregates 100+ cancer types. Cancer-specific conclusions require Cancer InFocus analysis.

7. **SVI as comparator.** Results are specific to SVI as the benchmark; other socioeconomic indices may show different divergence patterns.

8. **SHAP on StandardScaler space.** Mean |SHAP| values are not directly comparable across features with different raw distributions. These are relative within-model rankings, not absolute effect sizes.

9. **Cross-sectional design.** Causal inference is not possible from this study design.

10. **PCA single-component CHRI.** CHRI uses only PC1 (cardiometabolic); PC2 (cancer geography) is not incorporated. A two-component CHRI may better represent cancer-burden communities.

11. **Obesity model transferability.** R²=0.617 — lowest of all 10 diseases. Obesity-related CHRI contributions may be less reliable in states with unusual built or food environments.

12. **PC1 sign convention.** Fixed so positive scores = higher burden. Implementations should verify sign alignment.

---

## 6. Conclusion

Chronic disease prevention programs in the United States rely on the Social Vulnerability Index as the primary geographic targeting tool, but SVI quantifies socioeconomic deprivation, not actual disease burden. This study demonstrates that between 760 and 1,907 US census tracts — representing 3 to 7.6 million Americans — carry elevated chronic disease burden while ranking in the lowest quintile of socioeconomic vulnerability by SVI. These are predominantly aging communities (median tract age 56.2 years) in the pre-mortality-peak phase, where accumulated cardiometabolic and cancer burden has not yet produced excess mortality (life expectancy 79.5 years, near-national-average). This interval is precisely the window in which preventive chronic disease programs have maximum leverage.

CHRI provides a methodological foundation for identifying this population directly — from co-occurring disease rates validated against independent life expectancy data — rather than through socioeconomic proxies. The state-holdout validated XGBoost pipeline achieves R²=0.934–0.966 for cardiometabolic diseases across states entirely withheld from training, demonstrating that geographic disease patterns are transferable nationally. SHAP explainability reveals that cardiometabolic geography and cancer geography are structurally distinct phenomena — separated on PC1 and PC2 respectively — with implications for how prevention programs should be designed differently for each disease domain.

CHRI is designed to complement — not replace — SVI-based targeting in jurisdictions that wish to identify disease-burden-defined need independently of socioeconomic proxies. The 760–1,907 communities identified here represent a testable, geographically specific hypothesis: that chronic disease intervention programs targeting these tracts would reach populations with high preventable burden and sufficient life expectancy to benefit from preventive care. Prospective evaluation of program outcomes in divergent versus SVI-matched control communities would directly test this hypothesis and constitute the logical clinical extension of this work.

---

## Citation Placeholders (to fill before submission)
1. CDC chronic disease overview — cdc.gov/chronicdisease (or MMWR)
2. Flanagan 2011 — J Homeland Security Emerg Mgmt, DOI: 10.2202/1547-7355.1792
3. Jiao & Feng 2021 — Computers, Environment and Urban Systems, Vol. 85
4. Prior SVI-burden divergence paper — needs lit search
