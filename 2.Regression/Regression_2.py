# -------------------------------------------------------------
# REGRESSION MODEL WITH GDP_PPP CONSTANT 2021, FIXED VS. NO FIXED EFFECTS,
# AND ROBUSTNESS CHECKS BY TECHNOLOGY & REGION
# -------------------------------------------------------------
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 0. Load dataset
# -------------------------------------------------------------
df = pd.read_csv("/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data1/wacc_with_gdpppp2021.csv")
df.rename(columns={"GDP_PPP": "gdp_ppp"}, inplace=True)

# -------------------------------------------------------------
# 1. Add Region from ISO
# -------------------------------------------------------------
iso_to_region = {
    'ARG': 'Latin America & Caribbean', 'AUS': 'East Asia & Pacific', 'AUT': 'Europe & Central Asia',
    'BEL': 'Europe & Central Asia', 'BGD': 'South Asia', 'BGR': 'Europe & Central Asia',
    'BHR': 'Middle East & North Africa', 'BOL': 'Latin America & Caribbean', 'BRA': 'Latin America & Caribbean',
    'BWA': 'Sub-Saharan Africa', 'CHE': 'Europe & Central Asia', 'CHL': 'Latin America & Caribbean',
    'CHN': 'East Asia & Pacific', 'CMR': 'Sub-Saharan Africa', 'COL': 'Latin America & Caribbean',
    'CRI': 'Latin America & Caribbean', 'CYP': 'Europe & Central Asia', 'CZE': 'Europe & Central Asia',
    'DEU': 'Europe & Central Asia', 'DNK': 'Europe & Central Asia', 'EGY': 'Middle East & North Africa',
    'ESP': 'Europe & Central Asia', 'EST': 'Europe & Central Asia', 'FIN': 'Europe & Central Asia',
    'FRA': 'Europe & Central Asia', 'GBR': 'Europe & Central Asia', 'GHA': 'Sub-Saharan Africa',
    'GRC': 'Europe & Central Asia', 'GTM': 'Latin America & Caribbean', 'HRV': 'Europe & Central Asia',
    'HUN': 'Europe & Central Asia', 'IDN': 'East Asia & Pacific', 'IND': 'South Asia', 'IRL': 'Europe & Central Asia',
    'ISR': 'Middle East & North Africa', 'ITA': 'Europe & Central Asia', 'JAM': 'Latin America & Caribbean',
    'JOR': 'Middle East & North Africa', 'KEN': 'Sub-Saharan Africa', 'KHM': 'East Asia & Pacific',
    'LKA': 'South Asia', 'LTU': 'Europe & Central Asia', 'LVA': 'Europe & Central Asia',
    'MAR': 'Middle East & North Africa', 'MEX': 'Latin America & Caribbean', 'MLT': 'Middle East & North Africa',
    'MOZ': 'Sub-Saharan Africa', 'MUS': 'Sub-Saharan Africa', 'MYS': 'East Asia & Pacific', 'NAM': 'Sub-Saharan Africa',
    'NGA': 'Sub-Saharan Africa', 'NLD': 'Europe & Central Asia', 'NOR': 'Europe & Central Asia',
    'OMN': 'Middle East & North Africa', 'PAK': 'South Asia', 'PAN': 'Latin America & Caribbean',
    'PER': 'Latin America & Caribbean', 'PHL': 'East Asia & Pacific', 'POL': 'Europe & Central Asia',
    'PRT': 'Europe & Central Asia', 'PRY': 'Latin America & Caribbean', 'ROU': 'Europe & Central Asia',
    'SAU': 'Middle East & North Africa', 'SEN': 'Sub-Saharan Africa', 'SGP': 'East Asia & Pacific',
    'SLV': 'Latin America & Caribbean', 'SVK': 'Europe & Central Asia', 'SVN': 'Europe & Central Asia',
    'SWE': 'Europe & Central Asia', 'TUN': 'Middle East & North Africa', 'TUR': 'Europe & Central Asia',
    'TWN': 'East Asia & Pacific', 'TZA': 'Sub-Saharan Africa', 'UGA': 'Sub-Saharan Africa',
    'UKR': 'Europe & Central Asia', 'URY': 'Latin America & Caribbean', 'USA': 'North America',
    'VEN': 'Latin America & Caribbean', 'VNM': 'East Asia & Pacific', 'YEM': 'Middle East & North Africa',
    'ZAF': 'Sub-Saharan Africa', 'ZMB': 'Sub-Saharan Africa'
}
df["Region"] = df["ISO"].map(iso_to_region)

# -------------------------------------------------------------
# 2. Technology dummies
# -------------------------------------------------------------
df["is_solar"] = df["technology"].str.contains("solar", case=False, na=False).astype(int)
df["is_wind_onshore"] = df["technology"].str.contains("onshore", case=False, na=False).astype(int)
df["is_wind_offshore"] = df["technology"].str.contains("offshore", case=False, na=False).astype(int)

# -------------------------------------------------------------
# 3. Standardize macro variables
# -------------------------------------------------------------
df = df.replace([float('inf'), float('-inf')], pd.NA)
continuous = ["gdp_ppp", "inflation", "unemployment"]
scaler = StandardScaler()
df[continuous] = scaler.fit_transform(df[continuous])

# -------------------------------------------------------------
# 4. Base regressions: pooled OLS & FE
# -------------------------------------------------------------
base_sets = {
    "Onshore_ref": ["is_solar", "is_wind_offshore"],
    "Solar_ref": ["is_wind_onshore", "is_wind_offshore"]
}
results = []

for name, tech_dums in base_sets.items():
    predictors = continuous + tech_dums
    tmp = df.dropna(subset=predictors + ["wacc", "ISO", "Year"]).copy()

    X_ols = sm.add_constant(tmp[predictors])
    y_ols = tmp["wacc"]
    ols = sm.OLS(y_ols, X_ols).fit(cov_type="cluster", cov_kwds={"groups": tmp["ISO"]})

    panel_data = tmp.set_index(["ISO", "Year"])
    fe_model = PanelOLS(
        panel_data["wacc"], panel_data[predictors],
        entity_effects=True, time_effects=True,
        check_rank=False, drop_absorbed=True
    )
    fe = fe_model.fit(cov_type="clustered", cluster_entity=True)

    results.append({
        "Reference": name,
        "OLS_adj_R2": ols.rsquared_adj,
        "FE_within_R2": fe.rsquared_within,
        "OLS_params": ols.params,
        "FE_params": fe.params
    })

# -------------------------------------------------------------
# 5. Summary display
# -------------------------------------------------------------
for res in results:
    print(f"\n=== {res['Reference']} ===")
    print(f"Adjusted R² (OLS):        {res['OLS_adj_R2']:.4f}")
    print(f"Within R² (Fixed Effects): {res['FE_within_R2']:.4f}")
    print("\nTechnology Coefficients (OLS):")
    print(res["OLS_params"].filter(like="is_"))
    print("\nTechnology Coefficients (FE):")
    print(res["FE_params"].filter(like="is_"))

# -------------------------------------------------------------
# 6. Robustness: Technology & Region
# -------------------------------------------------------------
tech_labels = {
    "Solar_PV": {"is_solar": 1},
    "Wind_Onshore": {"is_wind_onshore": 1},
    "Wind_Offshore": {"is_wind_offshore": 1}
}
regions = df["Region"].dropna().unique()
robustness_results = []

print("\n\n================= ROBUSTNESS: BY TECHNOLOGY =================\n")
for label, tech_filter in tech_labels.items():
    subset = df.copy()
    for col, val in tech_filter.items():
        subset = subset[subset[col] == val]
    if subset.shape[0] < 20:
        print(f"⚠️ Not enough data for {label} — skipped.")
        continue
    predictors = ["gdp_ppp", "inflation", "unemployment"]
    subset = subset.dropna(subset=predictors + ["wacc", "ISO", "Year"])
    X = sm.add_constant(subset[predictors])
    y = subset["wacc"]
    ols = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": subset["ISO"]})
    panel = subset.set_index(["ISO", "Year"])
    fe_model = PanelOLS(
        panel["wacc"], panel[predictors],
        entity_effects=True, time_effects=True,
        check_rank=False, drop_absorbed=True
    )
    fe = fe_model.fit(cov_type="clustered", cluster_entity=True)
    print(f"\n▶ Technology: {label}")
    print(f"  OLS Adj. R²: {ols.rsquared_adj:.4f}")
    print(f"  FE Within R²: {fe.rsquared_within:.4f}")
    robustness_results.append({
        "Group": label, "Type": "Technology",
        "OLS_R2": ols.rsquared_adj, "FE_R2": fe.rsquared_within
    })

print("\n\n================= ROBUSTNESS: BY REGION =================\n")
for region in sorted(regions):
    subset = df[df["Region"] == region].copy()
    predictors = ["gdp_ppp", "inflation", "unemployment", "is_solar", "is_wind_offshore"]
    subset = subset.dropna(subset=predictors + ["wacc", "ISO", "Year"])
    if subset.shape[0] < 20:
        print(f"⚠️ Not enough data for {region} — skipped.")
        continue
    X = sm.add_constant(subset[predictors])
    y = subset["wacc"]
    ols = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": subset["ISO"]})
    panel = subset.set_index(["ISO", "Year"])
    fe_model = PanelOLS(
        panel["wacc"], panel[predictors],
        entity_effects=True, time_effects=True,
        check_rank=False, drop_absorbed=True
    )
    fe = fe_model.fit(cov_type="clustered", cluster_entity=True)
    print(f"\n▶ Region: {region}")
    print(f"  OLS Adj. R²: {ols.rsquared_adj:.4f}")
    print(f"  FE Within R²: {fe.rsquared_within:.4f}")
    robustness_results.append({
        "Group": region, "Type": "Region",
        "OLS_R2": ols.rsquared_adj, "FE_R2": fe.rsquared_within
    })

# -------------------------------------------------------------
# 7. Save & Plot
# -------------------------------------------------------------
robust_df = pd.DataFrame(robustness_results)
robust_df.to_csv("robustness_check_results.csv", index=False)
print("\n✅ Saved to: robustness_check_results.csv")

for subset in ["Technology", "Region"]:
    df_sub = robust_df[robust_df["Type"] == subset].sort_values("FE_R2", ascending=False)
    ax = df_sub.plot(
        x="Group", y=["OLS_R2", "FE_R2"], kind="bar", figsize=(12, 6),
        title=f"Robustness Check — {subset} Subsets", rot=45, width=0.75
    )
    ax.set_ylabel("R²")
    ax.set_xlabel(f"{subset} Group")
    ax.legend(["Pooled OLS (Adj R²)", "Fixed Effects (Within R²)"])
    plt.tight_layout()
    plt.savefig(f"robustness_plot_{subset.lower()}.png", dpi=300)
    print(f"📊 Plot saved: robustness_plot_{subset.lower()}.png")

plt.show()
