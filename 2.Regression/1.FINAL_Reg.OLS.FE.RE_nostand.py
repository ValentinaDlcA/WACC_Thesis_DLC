# ================================================================
# Full Code: OLS, Fixed Effects (FE with and without population),
# and Random Effects (RE) Without Standardization
# ================================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
from scipy.stats import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(
    "/Users/valentinadlc/Documents/MASTER/MASTER THESIS/WACC_Thesis_DLC/1.Cleaning_and_Merges/final_wacc_macro_historical.csv"
)

# Create technology dummies
df["is_solar"] = df["technology"].str.contains("solar", case=False, na=False).astype(int)
df["is_wind_onshore"] = df["technology"].str.contains("onshore", case=False, na=False).astype(int)
df["is_wind_offshore"] = df["technology"].str.contains("offshore", case=False, na=False).astype(int)

# Replace inf with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# ---------------------------------------------------------------
# 2. Define predictor sets
# ---------------------------------------------------------------
continuous_all = ["gdp_ppp", "population", "inflation", "unemployment"]
continuous_no_pop = ["gdp_ppp", "inflation", "unemployment"]
tech_vars = ["is_solar", "is_wind_offshore"]

predictors_all = continuous_all + tech_vars
predictors_no_pop = continuous_no_pop + tech_vars

# ---------------------------------------------------------------
# 3. Clean data
# ---------------------------------------------------------------
df_clean_all = df.dropna(subset=predictors_all + ["wacc", "ISO", "Year"]).copy()
df_clean_nopop = df.dropna(subset=predictors_no_pop + ["wacc", "ISO", "Year"]).copy()

# ---------------------------------------------------------------
# 4. OLS (with population)
# ---------------------------------------------------------------
X_ols = sm.add_constant(df_clean_all[predictors_all])
y = df_clean_all["wacc"]
ols = sm.OLS(y, X_ols).fit(cov_type="cluster", cov_kwds={"groups": df_clean_all["ISO"]})

# ---------------------------------------------------------------
# 5. FE (with population)
# ---------------------------------------------------------------
panel_fe_all = df_clean_all.set_index(["ISO", "Year"])
fe_all = PanelOLS(panel_fe_all["wacc"], panel_fe_all[predictors_all],
                  entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)

# ---------------------------------------------------------------
# 6. FE (without population)
# ---------------------------------------------------------------
panel_fe_nopop = df_clean_nopop.set_index(["ISO", "Year"])
fe_nopop = PanelOLS(panel_fe_nopop["wacc"], panel_fe_nopop[predictors_no_pop],
                    entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)

# ---------------------------------------------------------------
# 7. Random Effects (with population)
# ---------------------------------------------------------------
panel_re = df_clean_all.set_index(["ISO", "Year"])
re = RandomEffects(panel_re["wacc"], panel_re[predictors_all]).fit()

# ---------------------------------------------------------------
# 8. Hausman Test
# ---------------------------------------------------------------
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

stat, pval = hausman(fe_all, re)

# ---------------------------------------------------------------
# 9. Diagnostics
# ---------------------------------------------------------------
print("\n--- Variance Inflation Factors (VIF) ---")
vif_data = pd.DataFrame()
vif_data["Variable"] = predictors_all
vif_data["VIF"] = [variance_inflation_factor(df_clean_all[predictors_all].values, i)
                   for i in range(len(predictors_all))]
print(vif_data)

bp_test = het_breuschpagan(ols.resid, ols.model.exog)
print("\n--- Breusch-Pagan test for heteroskedasticity ---")
print(f"LM statistic: {bp_test[0]:.4f}, p-value: {bp_test[1]:.4f}")

dw_stat = durbin_watson(ols.resid)
print("\n--- Durbin-Watson test for autocorrelation ---")
print(f"Durbin-Watson statistic: {dw_stat:.4f} (value near 2 suggests no autocorrelation)")

# ---------------------------------------------------------------
# 10. Output summaries
# ---------------------------------------------------------------
print("\n=== Model Comparison ===")
print(f"OLS Adj. R²:        {ols.rsquared_adj:.4f}")
print(f"FE (with pop) R²:   {fe_all.rsquared_within:.4f}")
print(f"FE (no pop) R²:     {fe_nopop.rsquared_within:.4f}")
print(f"RE Overall R²:      {re.rsquared:.4f}")
print(f"Hausman p-value:    {pval:.4f}")

print("\n=== OLS Coefficients ===")
print(ols.summary())

print("\n=== Fixed Effects (WITH population) ===")
print(fe_all.summary)

print("\n=== Fixed Effects (WITHOUT population) ===")
print(fe_nopop.summary)

print("\n=== Random Effects Coefficients ===")
print(re.summary)

if pval < 0.05:
    print("\nRecommendation: Use Fixed Effects model (reject RE assumptions).")
else:
    print("\nRecommendation: Use Random Effects model (RE preferred).")
