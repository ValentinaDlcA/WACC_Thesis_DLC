# -------------------------------------------------------------
# REGRESSION MODEL: OLS, FE, RE + HAUSMAN TEST + CORRELATION + ROBUSTNESS
# -------------------------------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2

# -------------------------------------------------------------
# 0. Load and clean
# -------------------------------------------------------------
df = pd.read_csv("/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data1/wacc_with_gdpppp2021.csv")
df.rename(columns={"GDP_PPP": "gdp_ppp"}, inplace=True)
df = df.replace([np.inf, -np.inf], np.nan)

# Add region mapping
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

# Technology dummies
df["is_solar"] = df["technology"].str.contains("solar", case=False, na=False).astype(int)
df["is_wind_onshore"] = df["technology"].str.contains("onshore", case=False, na=False).astype(int)
df["is_wind_offshore"] = df["technology"].str.contains("offshore", case=False, na=False).astype(int)

# Standardize macro vars
df = df.replace([np.inf, -np.inf], np.nan)
continuous = ["gdp_ppp", "inflation", "unemployment"]
scaler = StandardScaler()
df[continuous] = scaler.fit_transform(df[continuous])

# -------------------------------------------------------------
# 1. Exploratory Correlation Matrix
# -------------------------------------------------------------
corr_vars = ["wacc"] + continuous + ["is_solar", "is_wind_onshore", "is_wind_offshore"]
corr_matrix = df[corr_vars].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix with WACC")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 2. Hausman Test function
# -------------------------------------------------------------
def hausman(fe, re):
    b = fe.params
    B = re.params
    common = b.index.intersection(B.index)
    b = b[common]
    B = B[common]
    v_b = fe.cov.loc[common, common]
    v_B = re.cov.loc[common, common]
    stat = np.dot((b - B).T, np.linalg.inv(v_b - v_B)).dot(b - B)
    pval = chi2.sf(stat, df=len(b))
    return stat, pval

# -------------------------------------------------------------
# 3. Base model estimation and comparison
# -------------------------------------------------------------
base_sets = {
    "Onshore_ref": ["is_solar", "is_wind_offshore"],
    "Solar_ref": ["is_wind_onshore", "is_wind_offshore"]
}
results = []

for name, tech_dums in base_sets.items():
    predictors = continuous + tech_dums
    tmp = df.dropna(subset=predictors + ["wacc", "ISO", "Year"]).copy()
    X = sm.add_constant(tmp[predictors])
    y = tmp["wacc"]

    ols = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": tmp["ISO"]})
    panel = tmp.set_index(["ISO", "Year"])
    fe = PanelOLS(panel["wacc"], panel[predictors],
                  entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    re = RandomEffects(panel["wacc"], panel[predictors]).fit()
    stat, pval = hausman(fe, re)

    print(f"\n=== {name} ===")
    print(f"OLS Adj. R²:     {ols.rsquared_adj:.4f}")
    print(f"FE Within R²:    {fe.rsquared_within:.4f}")
    print(f"Hausman p-value: {pval:.4f}")
    print("OLS Coefs:\n", ols.params.filter(like="is_"))
    print("FE Coefs:\n", fe.params.filter(like="is_"))
    print("RE Coefs:\n", re.params.filter(like="is_"))

# Optional: export correlations if needed
corr_matrix.to_csv("correlation_matrix.csv")
print("✅ Correlation matrix saved as CSV.")

# 4. Robustness Checks by Technology & Region
# -------------------------------------------------------------
robust_results = []
tech_labels = {
    "Solar_PV": {"is_solar": 1},
    "Wind_Onshore": {"is_wind_onshore": 1},
    "Wind_Offshore": {"is_wind_offshore": 1}
}
regions = df["Region"].dropna().unique()

def run_model_subset(subset, predictors, label, typ):
    X = sm.add_constant(subset[predictors])
    y = subset["wacc"]
    ols = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": subset["ISO"]})
    panel = subset.set_index(["ISO", "Year"])
    fe = PanelOLS(panel["wacc"], panel[predictors],
                  entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    re = RandomEffects(panel["wacc"], panel[predictors]).fit()
    stat, pval = hausman(fe, re)

    print(f"\n▶ {typ}: {label}")
    print(f"  OLS Adj. R²:     {ols.rsquared_adj:.4f}")
    print(f"  FE Within R²:    {fe.rsquared_within:.4f}")
    print(f"  Hausman p-val:   {pval:.4f}")

    robust_results.append({
        "Group": label, "Type": typ,
        "OLS_R2": ols.rsquared_adj,
        "FE_R2": fe.rsquared_within,
        "Hausman_pval": pval
    })

print("\n\n================= ROBUSTNESS: TECHNOLOGY =================")
for label, filter_ in tech_labels.items():
    sub = df.copy()
    for col, val in filter_.items():
        sub = sub[sub[col] == val]
    sub = sub.dropna(subset=continuous + ["wacc", "ISO", "Year"])
    if len(sub) >= 30:
        run_model_subset(sub, continuous, label, "Technology")

print("\n\n================= ROBUSTNESS: REGION =================")
for region in sorted(regions):
    sub = df[df["Region"] == region].dropna(subset=continuous + ["wacc", "ISO", "Year"])
    if len(sub) >= 30:
        run_model_subset(sub, continuous + ["is_solar", "is_wind_offshore"], region, "Region")

# -------------------------------------------------------------
# 5. Save and Plot
# -------------------------------------------------------------
robust_df = pd.DataFrame(robust_results)
robust_df.to_csv("robustness_check_results.csv", index=False)
print("\n✅ Robustness results saved.")

for group in ["Technology", "Region"]:
    df_plot = robust_df[robust_df["Type"] == group].sort_values("FE_R2", ascending=False)
    ax = df_plot.plot(
        x="Group", y=["OLS_R2", "FE_R2"], kind="bar", figsize=(12, 6),
        title=f"Robustness Check — {group}", rot=45
    )
    ax.set_ylabel("R²")
    plt.tight_layout()
    plt.savefig(f"robustness_plot_{group.lower()}.png", dpi=300)
    print(f"📊 Plot saved: robustness_plot_{group.lower()}.png")

plt.show()
