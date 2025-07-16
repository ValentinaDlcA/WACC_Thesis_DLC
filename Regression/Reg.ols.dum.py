import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# Load data
df = pd.read_csv(
    "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data1/wacc_with_gdpppp2021.csv"
)
df.rename(columns={"GDP_PPP": "gdp_ppp"}, inplace=True)

# Technology dummies
df["is_solar"] = df["technology"].str.contains("solar", case=False, na=False).astype(int)
df["is_wind_onshore"] = df["technology"].str.contains("onshore", case=False, na=False).astype(int)
df["is_wind_offshore"] = df["technology"].str.contains("offshore", case=False, na=False).astype(int)

# Standardize macro vars INCLUDING population
df = df.replace([np.inf, -np.inf], np.nan)
continuous = ["gdp_ppp", "population", "inflation", "unemployment"]
df[continuous] = StandardScaler().fit_transform(df[continuous])

# Create explicit year dummies for RE model (already done)
year_dummies = pd.get_dummies(df["Year"], prefix="Year", drop_first=True)
df = pd.concat([df, year_dummies], axis=1)

# Create explicit country dummies for OLS model
country_dummies = pd.get_dummies(df["ISO"], prefix="Country", drop_first=True)
df = pd.concat([df, country_dummies], axis=1)

# Hausman test function
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

# Base predictors (tech + macro vars)
tech_vars = ["is_solar", "is_wind_offshore"]
time_vars = list(year_dummies.columns)  # All time dummies except first year
country_vars = list(country_dummies.columns)  # All country dummies except first country

# OLS predictors include macro, tech, year dummies, country dummies
predictors_ols = continuous + tech_vars + time_vars + country_vars

# FE and RE predictors do NOT include country dummies (handled internally)
predictors_ols_fe = continuous + tech_vars
predictors_re = continuous + tech_vars + time_vars  # Include time dummies for RE

# Clean data for OLS (with all dummies)
df_clean_ols = df.dropna(subset=predictors_ols + ["wacc"]).copy()
X_ols = sm.add_constant(df_clean_ols[predictors_ols])
y_ols = df_clean_ols["wacc"]

# OLS with year & country fixed effects as dummies
ols = sm.OLS(y_ols, X_ols).fit(cov_type="cluster", cov_kwds={"groups": df_clean_ols["ISO"]})

# Clean data for FE and RE (no country dummies here)
df_clean = df.dropna(subset=predictors_ols_fe + ["wacc", "ISO", "Year"]).copy()

# Panel data for FE and RE
panel_data_fe = df_clean.set_index(["ISO", "Year"])

# Fixed Effects model (entity and time effects)
fe = PanelOLS(panel_data_fe["wacc"], panel_data_fe[predictors_ols_fe],
              entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)

# For RE, must include time dummies explicitly
df_clean_re = df_clean.dropna(subset=predictors_re + ["wacc", "ISO", "Year"]).copy()
panel_data_re = df_clean_re.set_index(["ISO", "Year"])
X_re = panel_data_re[predictors_re]

re = RandomEffects(panel_data_re["wacc"], X_re).fit()

# Diagnostic Tests for OLS
print("\n--- Variance Inflation Factors (VIF) ---")
vif_data = pd.DataFrame()
vif_data["Variable"] = predictors_ols
vif_data["VIF"] = [variance_inflation_factor(df_clean_ols[predictors_ols].values, i) for i in range(len(predictors_ols))]
print(vif_data)

bp_test = het_breuschpagan(ols.resid, ols.model.exog)
print("\n--- Breusch-Pagan test for heteroskedasticity ---")
print(f"LM statistic: {bp_test[0]:.4f}, p-value: {bp_test[1]:.4f}")

dw_stat = durbin_watson(ols.resid)
print("\n--- Durbin-Watson test for autocorrelation ---")
print(f"Durbin-Watson statistic: {dw_stat:.4f} (value near 2 suggests no autocorrelation)")

# Hausman test
stat, pval = hausman(fe, re)

# Output results
print("\n=== Model Comparison ===")
print(f"OLS Adj. R²:      {ols.rsquared_adj:.4f}")
print(f"FE Within R²:     {fe.rsquared_within:.4f}")
print(f"RE Overall R²:    {re.rsquared:.4f}")
print(f"Hausman test p-value: {pval:.4f}")

print("\n=== OLS Coefficients ===")
print(ols.summary())

print("\n=== Fixed Effects Coefficients ===")
print(fe.summary)

print("\n=== Random Effects Coefficients ===")
print(re.summary)

if pval < 0.05:
    print("\nRecommendation: Use Fixed Effects model (reject RE assumptions).")
else:
    print("\nRecommendation: Use Random Effects model (RE preferred).")
