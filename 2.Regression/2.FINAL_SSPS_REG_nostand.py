# REGRESSION SSPS - WACC with Technology Dummies
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
import seaborn as sns

# === Load merged dataset ===
data_path = "/Users/valentinadlc/Documents/MASTER/MASTER THESIS/WACC_Thesis_DLC/1.Cleaning_and_Merges/final_merged_dataset.csv"
df = pd.read_csv(data_path).dropna()

# === Ensure index and dependent variable ===
df = df.set_index(['Country', 'Year'])
y = df['wacc']

# === Create technology dummies if not already present ===
# Only uncomment if they need to be created from 'technology' column:
# df['is_solar'] = df['technology'].str.contains('solar', case=False, na=False).astype(int)
# df['is_wind_offshore'] = df['technology'].str.contains('offshore', case=False, na=False).astype(int)

# === Model configurations ===
models = {
    'Urban only': ['Urban_Share', 'is_solar', 'is_wind_offshore'],
    'Urban + Governance': ['Urban_Share', 'Governance_Index', 'is_solar', 'is_wind_offshore'],
    'Urban + Rule of Law': ['Urban_Share', 'Rule_of_Law', 'is_solar', 'is_wind_offshore'],
}

# === Run regressions ===
for name, vars in models.items():
    print(f"\n\n================== {name.upper()} ==================")

    X = df[vars]
    X_ols = sm.add_constant(X)

    # --- Pooled OLS ---
    print("\n--- Pooled OLS ---")
    pooled_model = sm.OLS(y, X_ols).fit()
    print(pooled_model.summary())

    # --- Fixed Effects Model ---
    print("\n--- Fixed Effects (Robust SEs) ---")
    fe_model = PanelOLS(y, X, entity_effects=True)
    fe_results = fe_model.fit(cov_type='robust')
    print(fe_results.summary)

    # --- VIF (only if >1 predictor) ---
    if len(vars) > 1:
        print("\n--- Variance Inflation Factors (VIF) ---")
        vif_df = pd.DataFrame()
        vif_df["Variable"] = X.columns
        vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print(vif_df)

    # --- Breusch-Pagan Test (Heteroskedasticity) ---
    print("\n--- Breusch-Pagan Test ---")
    bp_test = het_breuschpagan(pooled_model.resid, X_ols)
    bp_labels = ['LM statistic', 'p-value', 'f-value', 'f p-value']
    print(dict(zip(bp_labels, bp_test)))

# === Correlation Heatmap for Visual Reference ===
print("\n=== Correlation Heatmap (Full Variables) ===")
sns.heatmap(df[['Urban_Share', 'Governance_Index', 'Rule_of_Law']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
