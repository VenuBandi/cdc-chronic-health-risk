"""
Phase 1 Analyses — CDC PLACES CHRI Manuscript
Runs 4 analyses required before writing draft7.

Usage:
    python3 phase1_analyses.py

Outputs saved to:
    ../outputs/phase1_sensitivity_table.csv
    ../outputs/phase1_subgroup_le_correlation.csv
    ../outputs/phase1_age_test.csv
    ../outputs/phase1_missing_tracts.csv
    ../outputs/phase1_summary_report.txt
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

# ── Checkpoint system ────────────────────────────────────────────────────────
CHECKPOINT_FILE = f"/Users/venubandi/Desktop/cdc places research/outputs/phase1_checkpoint.json"

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}

def _jsonify(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

def save_checkpoint(state: dict):
    existing = load_checkpoint()
    existing.update(_jsonify(state))
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  ✓ Checkpoint saved: {list(state.keys())}")

checkpoint = load_checkpoint()
if checkpoint:
    print(f"Resuming from checkpoint. Completed: {[k for k, v in checkpoint.items() if v]}")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE        = "/Users/venubandi/Desktop/cdc places research"
PLACES_PATH = "/Users/venubandi/PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2024_release_20251010 (1).csv"
DIVERG_PATH = f"{BASE}/outputs/svi_chri_divergence.csv"
AGE_PATH    = f"{BASE}/outputs/acs_median_age_tracts.csv"
OUT_DIR     = f"{BASE}/outputs"

# USALEEP — download if not found (CDC NCHS, public domain)
# If you already have it, set this path manually
USALEEP_PATH = f"{OUT_DIR}/usaleep_tract_le.csv"

report_lines = []

def log(msg):
    print(msg)
    report_lines.append(msg)

log("=" * 65)
log("PHASE 1 ANALYSES — CDC PLACES CHRI")
log("=" * 65)


# ── Load divergence data ─────────────────────────────────────────────────────
log("\nLoading divergence data...")
div = pd.read_csv(DIVERG_PATH)
div["TractFIPS"] = div["TractFIPS"].astype(str).str.zfill(11)
log(f"  Tracts in divergence file: {len(div):,}")


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — 760-Tract Threshold Sensitivity (3 × 3 grid)
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "─" * 65)
log("ANALYSIS 1 — 760-Tract Threshold Sensitivity")
log("─" * 65)
log("Varying SVI 'low' threshold × CHRI 'high' threshold.")
log("Original: SVI_pct <= 0.20, CHRI_pct >= 0.80\n")

svi_cuts  = [0.20, 0.25, 0.33]   # bottom 20% / 25% / 33% SVI
chri_cuts = [0.75, 0.80, 0.85]   # top 25% / 20% / 15% CHRI

rows = []
for s in svi_cuts:
    for c in chri_cuts:
        mask = (div["SVI_pct"] <= s) & (div["CHRI_pct"] >= c)
        n = mask.sum()
        # Population estimate: mean US census tract ~4,000 adults
        pop_est = n * 4000
        rows.append({
            "SVI_threshold": f"≤{int(s*100)}th pct",
            "CHRI_threshold": f"≥{int(c*100)}th pct",
            "n_tracts": n,
            "pop_estimate": f"~{pop_est/1_000_000:.1f}M",
        })
        log(f"  SVI ≤ {s:.0%}  ×  CHRI ≥ {c:.0%}  →  {n:>5,} tracts  (~{pop_est/1e6:.1f}M people)")

sens_df = pd.DataFrame(rows)
sens_df.to_csv(f"{OUT_DIR}/phase1_sensitivity_table.csv", index=False)

# Original cell: SVI_pct <= 0.20, CHRI_pct >= 0.80
orig_mask = (div["SVI_pct"] <= 0.20) & (div["CHRI_pct"] >= 0.80)
n_orig = orig_mask.sum()
n_min  = min(r["n_tracts"] for r in rows)
n_max  = max(r["n_tracts"] for r in rows)
log(f"\n  Original count (<=20th / >=80th): {n_orig:,} tracts")
log(f"  Range across all 9 cells: {n_min:,} – {n_max:,} tracts")
if n_max / n_min < 2.5:
    log("  VERDICT: Finding is robust — count stays in a tight range.")
else:
    log("  CAUTION: Count varies >2.5×. Headline number needs qualification.")
log(f"\n  → Saved: phase1_sensitivity_table.csv")
save_checkpoint({"analysis_1_sensitivity": True,
                 "original_count": n_orig,
                 "range_min": n_min,
                 "range_max": n_max})


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Subgroup LE Correlation in Divergent Tracts
# (requires USALEEP data — downloads if not present)
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "─" * 65)
log("ANALYSIS 2 — Subgroup Life Expectancy Correlation")
log("─" * 65)

def download_usaleep(path):
    """Download USALEEP tract-level life expectancy from CDC NCHS."""
    import urllib.request, zipfile, io, tempfile
    URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NVSS/USALEEP/CSV/US_A.CSV"
    log(f"  Downloading USALEEP from CDC NCHS...")
    try:
        with urllib.request.urlopen(URL, timeout=60) as resp:
            content = resp.read()
        # Try parsing directly as CSV
        from io import StringIO
        df_le = pd.read_csv(StringIO(content.decode("latin-1")), low_memory=False)
        df_le.to_csv(path, index=False)
        log(f"  Downloaded: {len(df_le):,} rows → {path}")
        return df_le
    except Exception as e:
        log(f"  Download failed: {e}")
        return None

if os.path.exists(USALEEP_PATH):
    le_raw = pd.read_csv(USALEEP_PATH, low_memory=False)
    log(f"  USALEEP loaded: {len(le_raw):,} rows")
else:
    log("  USALEEP not found locally — attempting download...")
    le_raw = download_usaleep(USALEEP_PATH)

if le_raw is not None:
    # Standardize column names (USALEEP uses "Tract ID" and "e(0)")
    le_raw.columns = [c.strip() for c in le_raw.columns]

    # Find the FIPS and LE columns (name varies by release)
    fips_col = next((c for c in le_raw.columns
                     if "tract" in c.lower() or "fips" in c.lower() or "geoid" in c.lower()), None)
    le_col   = next((c for c in le_raw.columns
                     if "e(0)" in c.lower() or "life_exp" in c.lower() or "e0" in c.lower()), None)

    if fips_col and le_col:
        le = le_raw[[fips_col, le_col]].copy()
        le.columns = ["TractFIPS", "life_expectancy"]
        le["TractFIPS"] = le["TractFIPS"].astype(str).str.zfill(11)
        le = le.dropna(subset=["life_expectancy"])
        le["life_expectancy"] = pd.to_numeric(le["life_expectancy"], errors="coerce")
        le = le.dropna(subset=["life_expectancy"])
        log(f"  Usable LE tracts: {len(le):,}")

        # Join to divergence data
        merged = div.merge(le, on="TractFIPS", how="inner")
        log(f"  Merged tracts (CHRI + LE): {len(merged):,}")

        # Overall correlation
        r_all, p_all = stats.pearsonr(merged["CHRI_pct"], merged["life_expectancy"])
        log(f"\n  Overall: r(CHRI_pct, LE) = {r_all:.3f}  (p={p_all:.2e})  n={len(merged):,}")

        # Subgroup: SVI-low / CHRI-high (divergent tracts)
        div_mask = (merged["SVI_pct"] <= 0.20) & (merged["CHRI_pct"] >= 0.80)
        div_sub  = merged[div_mask]
        if len(div_sub) >= 10:
            r_div, p_div = stats.pearsonr(div_sub["CHRI_pct"], div_sub["life_expectancy"])
            log(f"  SVI-low/CHRI-high subgroup: r = {r_div:.3f}  (p={p_div:.2e})  n={len(div_sub):,}")
            if abs(r_div) > 0.3:
                log("  VERDICT: CHRI tracks LE even in SVI-low tracts — 'missed' claim is validated.")
            else:
                log("  CAUTION: Weak correlation in divergent subgroup — qualify the 'missed' claim.")
        else:
            log(f"  Too few divergent tracts with LE data ({len(div_sub)}) — skip subgroup correlation.")

        # Mean LE by divergence class
        log("\n  Mean life expectancy by divergence class:")
        le_by_class = merged.groupby("divergence_class")["life_expectancy"].agg(["mean","count"])
        for cls, row in le_by_class.iterrows():
            log(f"    {cls:<25} {row['mean']:.1f} yrs  (n={int(row['count']):,})")

        # Save
        result = merged[["TractFIPS", "CHRI_pct", "SVI_pct",
                          "divergence_class", "life_expectancy"]].copy()
        result.to_csv(f"{OUT_DIR}/phase1_subgroup_le_correlation.csv", index=False)
        log(f"\n  → Saved: phase1_subgroup_le_correlation.csv")
        save_checkpoint({"analysis_2_le_correlation": True,
                         "r_overall": round(r_all, 3),
                         "r_divergent_subgroup": round(r_div, 3) if len(div_sub) >= 10 else "n/a"})
    else:
        log(f"  Could not identify FIPS ({fips_col}) or LE ({le_col}) column in USALEEP.")
        log(f"  Columns found: {list(le_raw.columns)}")
        save_checkpoint({"analysis_2_le_correlation": "column_error"})
else:
    log("  SKIPPING Analysis 2 — USALEEP data unavailable.")
    log("  To run manually: download US_A.CSV from")
    log("  https://www.cdc.gov/nchs/nvss/usaleep/usaleep.html")
    log(f"  Save to: {USALEEP_PATH}")
    save_checkpoint({"analysis_2_le_correlation": "skipped_no_data"})


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Age Test on 760-Tract Finding
# "Does the finding hold in below-median-age tracts?"
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "─" * 65)
log("ANALYSIS 3 — Age Test on 760-Tract Finding")
log("─" * 65)
log("Does the SVI-low/CHRI-high finding appear even in younger tracts?")
log("(Rebuts: 'CHRI just identifies aging communities')\n")

age = pd.read_csv(AGE_PATH)
age["TractFIPS"] = age["TractFIPS_str"].astype(str).str.zfill(11)
age = age[["TractFIPS", "median_age"]].dropna()
log(f"  Median age data: {len(age):,} tracts")

div_age = div.merge(age, on="TractFIPS", how="inner")
log(f"  Merged (CHRI + age): {len(div_age):,} tracts")

national_median_age = div_age["median_age"].median()
log(f"  National median tract age: {national_median_age:.1f} years")

# Full 760-tract count
orig_mask_full = (div_age["SVI_pct"] <= 0.20) & (div_age["CHRI_pct"] >= 0.80)
n_full = orig_mask_full.sum()
log(f"  SVI-low/CHRI-high tracts in merged dataset: {n_full:,}")

# Below-median-age tracts only
below_median = div_age[div_age["median_age"] < national_median_age]
orig_mask_young = (below_median["SVI_pct"] <= 0.20) & (below_median["CHRI_pct"] >= 0.80)
n_young = orig_mask_young.sum()
pct_young = n_young / n_full * 100 if n_full > 0 else 0
log(f"  Divergent tracts in BELOW-median-age communities: {n_young:,} ({pct_young:.1f}% of total)")

# Age distribution of divergent vs all
diverg_tracts = div_age[orig_mask_full]
all_tracts    = div_age

log(f"\n  Median age in ALL tracts: {all_tracts['median_age'].median():.1f} yrs")
log(f"  Median age in SVI-low/CHRI-high tracts: {diverg_tracts['median_age'].median():.1f} yrs")
age_diff = diverg_tracts["median_age"].median() - all_tracts["median_age"].median()
log(f"  Divergent tracts are {age_diff:+.1f} yrs older than average")

if n_young >= 50:
    log(f"\n  VERDICT: {n_young:,} divergent tracts ({pct_young:.0f}%) are in below-median-age")
    log("  communities — finding is NOT purely driven by aging demographics.")
elif n_young >= 20:
    log(f"\n  MODERATE: {n_young:,} divergent tracts in young communities.")
    log("  Finding is real but age is a contributing factor. Disclose both.")
else:
    log(f"\n  CAUTION: Only {n_young:,} divergent tracts are in below-median-age communities.")
    log("  Age confounding may be substantial. Strengthen Limitation 12.")

# Save
age_result = div_age[["TractFIPS", "StateAbbr", "CHRI_pct", "SVI_pct",
                        "divergence_class", "median_age"]].copy()
age_result["below_median_age"] = age_result["median_age"] < national_median_age
age_result["is_divergent"] = (age_result["SVI_pct"] <= 0.20) & (age_result["CHRI_pct"] >= 0.80)
age_result.to_csv(f"{OUT_DIR}/phase1_age_test.csv", index=False)
log(f"\n  → Saved: phase1_age_test.csv")
save_checkpoint({"analysis_3_age_test": True,
                 "n_divergent_total": n_full,
                 "n_divergent_below_median_age": n_young,
                 "pct_divergent_young": round(pct_young, 1),
                 "median_age_all_tracts": round(national_median_age, 1),
                 "median_age_divergent_tracts": round(diverg_tracts["median_age"].median(), 1)})


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4 — Characterize 5,077 Missing Tracts
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "─" * 65)
log("ANALYSIS 4 — Characterize 5,077 Missing Tracts")
log("─" * 65)
log("Are excluded tracts systematically different from included tracts?")
log("(Non-random missingness = undercount of true gap)\n")

if not os.path.exists(PLACES_PATH):
    log(f"  PLACES data not found at: {PLACES_PATH}")
    log("  Skipping Analysis 4.")
else:
    places = pd.read_csv(PLACES_PATH, low_memory=False)
    for col in ["TotalPopulation", "TotalPop18plus"]:
        if col in places.columns and places[col].dtype == object:
            places[col] = pd.to_numeric(
                places[col].str.replace(",", "", regex=False), errors="coerce"
            )

    # Standardize FIPS
    if "LocationID" in places.columns:
        places["TractFIPS"] = places["LocationID"].astype(str).str.zfill(11)
    elif "TractFIPS" in places.columns:
        places["TractFIPS"] = places["TractFIPS"].astype(str).str.zfill(11)

    n_total = len(places)
    included_fips = set(div["TractFIPS"].unique())
    places["included"] = places["TractFIPS"].isin(included_fips)
    n_included = places["included"].sum()
    n_missing  = n_total - n_included
    log(f"  Total PLACES tracts: {n_total:,}")
    log(f"  In divergence file (included): {n_included:,}")
    log(f"  Missing / excluded: {n_missing:,} ({n_missing/n_total*100:.1f}%)")

    # Compare included vs excluded on key indicators
    # Look for rurality proxy (population density or low population)
    compare_cols = []
    for candidate in [
        "TotalPopulation", "TotalPop18plus",
        "PHLTH_CrudePrev",     # poor physical health days (high = worse health)
        "DEPRESSION_CrudePrev",
        "ARTHRITIS_CrudePrev",
        "DIABETES_CrudePrev",
        "OBESITY_CrudePrev",
        "BPHIGH_CrudePrev",
        "ACCESS2_CrudePrev",   # lacking health insurance
        "TEETHLOST_CrudePrev", # tooth loss — proxy for rural/low-access
        "LPA_CrudePrev",       # physical inactivity
    ]:
        if candidate in places.columns:
            compare_cols.append(candidate)

    log(f"\n  Comparing included vs excluded tracts on {len(compare_cols)} indicators:")
    log(f"  {'Column':<30} {'Included mean':>14} {'Excluded mean':>14} {'Diff %':>8}")
    log(f"  {'─'*30} {'─'*14} {'─'*14} {'─'*8}")

    results_miss = []
    for col in compare_cols:
        inc_vals = pd.to_numeric(places.loc[places["included"], col], errors="coerce").dropna()
        exc_vals = pd.to_numeric(places.loc[~places["included"], col], errors="coerce").dropna()
        if len(inc_vals) < 10 or len(exc_vals) < 10:
            continue
        inc_mean = inc_vals.mean()
        exc_mean = exc_vals.mean()
        diff_pct = (exc_mean - inc_mean) / inc_mean * 100 if inc_mean != 0 else 0
        _, pval = stats.mannwhitneyu(inc_vals, exc_vals, alternative="two-sided")
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        log(f"  {col:<30} {inc_mean:>14.2f} {exc_mean:>14.2f} {diff_pct:>+7.1f}% {sig}")
        results_miss.append({
            "column": col,
            "included_mean": round(inc_mean, 3),
            "excluded_mean": round(exc_mean, 3),
            "diff_pct": round(diff_pct, 1),
            "pval": round(pval, 4),
        })

    # Population size as rurality proxy
    if "TotalPop18plus" in places.columns:
        inc_pop = pd.to_numeric(places.loc[places["included"], "TotalPop18plus"],
                                errors="coerce").dropna()
        exc_pop = pd.to_numeric(places.loc[~places["included"], "TotalPop18plus"],
                                errors="coerce").dropna()
        log(f"\n  Median adult population:")
        log(f"    Included tracts: {inc_pop.median():,.0f}")
        log(f"    Excluded tracts: {exc_pop.median():,.0f}")
        if exc_pop.median() < inc_pop.median() * 0.5:
            log("  → Excluded tracts are substantially smaller (likely rural/tribal).")
        else:
            log("  → Population size is similar; missingness may not be rural-concentrated.")

    # State distribution of missing tracts
    if "StateAbbr" in places.columns:
        state_missing = (places[~places["included"]]
                         .groupby("StateAbbr").size()
                         .sort_values(ascending=False)
                         .head(10))
        log(f"\n  Top states by missing tract count:")
        for state, cnt in state_missing.items():
            log(f"    {state}: {cnt:,} missing tracts")

    miss_df = pd.DataFrame(results_miss)
    miss_df.to_csv(f"{OUT_DIR}/phase1_missing_tracts.csv", index=False)
    log(f"\n  → Saved: phase1_missing_tracts.csv")
    save_checkpoint({"analysis_4_missing_tracts": True,
                     "n_total_places": n_total,
                     "n_included": n_included,
                     "n_missing": n_missing,
                     "pct_missing": round(n_missing / n_total * 100, 1)})


# ── Save full report ──────────────────────────────────────────────────────────
report_path = f"{OUT_DIR}/phase1_summary_report.txt"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

log("\n" + "=" * 65)
log("ALL ANALYSES COMPLETE")
log(f"Summary saved: {report_path}")
log("=" * 65)
