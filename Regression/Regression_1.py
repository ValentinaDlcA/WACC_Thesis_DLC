# REGRESSION MODEL WITH NEW GDP_PPP CONSTANT 2021, FIXED VS. NO FIXED EFFECTS, ALL TESTS, AND COMPARED SOLAR VS. ONSHORE TECH AS REFERENCE-------------------------------------------------------------
# 0. imports & data
# -------------------------------------------------------------
import pandas as pd, statsmodels.api as sm
from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/"
    "demo/PyCharmLearningProject/data1/wacc_with_gdpppp2021.csv"
)

# technology flags
df["is_solar"]         = df["technology"].str.contains("solar",    case=False, na=False).astype(int)
df["is_wind_onshore"]  = df["technology"].str.contains("onshore",  case=False, na=False).astype(int)
df["is_wind_offshore"] = df["technology"].str.contains("offshore", case=False, na=False).astype(int)

continuous = ["GDP_PPP", "population", "inflation", "unemployment"]
scaler = StandardScaler()
df[continuous] = scaler.fit_transform(df[continuous])

base_sets = {
    "Onshore_ref":   ["is_solar", "is_wind_offshore"],   # on-shore dropped
    "Solar_ref":     ["is_wind_onshore", "is_wind_offshore"]  # solar dropped
}

results = {}

for name, tech_dums in base_sets.items():
    predictors = continuous + tech_dums
    tmp = df.dropna(subset=predictors + ["wacc", "ISO", "Year"]).copy()

    # ---------- pooled OLS ----------
    X = sm.add_constant(tmp[predictors])
    y = tmp["wacc"]
    ols = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": tmp["ISO"]})

    # ---------- FE: country + year ----------
    fe = PanelOLS(tmp.set_index(["ISO", "Year"])["wacc"],
                  tmp.set_index(["ISO", "Year"])[predictors],
                  entity_effects=True, time_effects=True
                 ).fit(cov_type="clustered", cluster_entity=True)

    results[name] = {"OLS": ols, "FE": fe}

# -------------------------------------------------------------
# Compare technology premiums
# -------------------------------------------------------------
comp = pd.DataFrame({
    "OLS_onshore_ref" : results["Onshore_ref"]["OLS"].params[["is_solar","is_wind_offshore"]],
    "OLS_solar_ref"   : results["Solar_ref"]["OLS"].params[["is_wind_onshore","is_wind_offshore"]],
    "FE_onshore_ref"  : results["Onshore_ref"]["FE"].params[["is_solar","is_wind_offshore"]],
    "FE_solar_ref"    : results["Solar_ref"]["FE"].params[["is_wind_onshore","is_wind_offshore"]],
})
print("\n=== Technology premiums under two reference choices ===")
print(comp)

# full summaries
print(results["Onshore_ref"]["OLS"].summary())
print(results["Solar_ref"]["OLS"].summary())
print(results["Onshore_ref"]["FE"].summary)
print(results["Solar_ref"]["FE"].summary)


