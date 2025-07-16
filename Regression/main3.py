# REGRESSION NO FIXED EFFECTS + FIXED EFFECTS-------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------------
# ==============================================================
# 0. Imports
# ==============================================================
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------
path = (
    "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/"
    "demo/PyCharmLearningProject/data1/wacc_with_gdpppp2021.csv"
)
df = pd.read_csv(path)

# --------------------------------------------------------------
# 2. Tech dummies (drop is_wind_onshore as reference)
# --------------------------------------------------------------
df["is_solar"]         = df["technology"].str.contains("solar",    na=False, case=False).astype(int)
df["is_wind_onshore"]  = df["technology"].str.contains("onshore",  na=False, case=False).astype(int)
df["is_wind_offshore"] = df["technology"].str.contains("offshore", na=False, case=False).astype(int)

# Keep only two dummies (solar & offshore); on-shore = reference
tech_dummies = ["is_solar", "is_wind_offshore"]

# --------------------------------------------------------------
# 3. Predictor list + scaling
# --------------------------------------------------------------
continuous = ["GDP_PPP", "population", "inflation", "unemployment"]
predictors = continuous + tech_dummies    # drop gdp & on-shore dummy

# Scale continuous vars
scaler = StandardScaler()
df[continuous] = scaler.fit_transform(df[continuous])

# Complete-case subset
reg_df = df.dropna(subset=predictors + ["wacc", "ISO", "Year"]).copy()

# --------------------------------------------------------------
# 4-A.  Pooled OLS  (robust SE)
# --------------------------------------------------------------
X = sm.add_constant(reg_df[predictors])
y = reg_df["wacc"]
ols = sm.OLS(y, X).fit()

print("\n================  POOLED OLS  (classical SE)  ================")
print(ols.summary())

# ------ Newey-West HAC (lag = 1 year) -------------------------
nw = ols.get_robustcov_results(cov_type="HAC", maxlags=1)
print("\n-----  OLS with Newey-West HAC (lag 1)  -----")
print(nw.summary())

# ------ Cluster-robust by ISO ---------------------------------
clusters = reg_df["ISO"]
cluster = ols.get_robustcov_results(cov_type="cluster", groups=clusters)
print("\n-----  OLS with Country-Clustered SE  -----")
print(cluster.summary())

# ------ VIF after dropping dummy trap -------------------------
vif_df = pd.DataFrame({
    "Var": X.columns,
    "VIF": [variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])]
})
print("\n=== VIF (after dropping on-shore dummy) ===")
print(vif_df)

# --------------------------------------------------------------
# 4-B.  Country & Year Fixed-Effects with robust SE
# --------------------------------------------------------------
panel_df = reg_df.set_index(["ISO", "Year"])
X_fe = panel_df[predictors]
y_fe = panel_df["wacc"]

fe_model  = PanelOLS(y_fe, X_fe, entity_effects=True, time_effects=True)
fe_result = fe_model.fit(cov_type="clustered", cluster_entity=True)

print("\n==========  Country & Year Fixed Effects  (cluster-robust) ==========")
print(fe_result.summary)

# --------------------------------------------------------------
# 5. Quick coefficient comparison
# --------------------------------------------------------------
compare = pd.DataFrame({
    "Variable" : ["Intercept"] + predictors,
    "OLS_coef" : [ols.params["const"]] + list(ols.params[predictors]),
    "FE_coef"  : [None] + list(fe_result.params[predictors])  # no global intercept in FE
})
print("\n=== Coefficient comparison (OLS vs FE) ===")
print(compare.to_string(index=False))
