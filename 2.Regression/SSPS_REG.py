# REGRESSION SSPS - WACC
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
import seaborn as sns

# === Load merged dataset ===
data_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/final_merged_dataset.csv"
df = pd.read_csv(data_path).dropna()
df = df.set_index(['Country', 'Year'])
y = df['wacc']

# === Model configurations ===
models = {
    'Urban + Governance': ['Urban_Share', 'Governance_Index'],
    'Urban + Rule of Law': ['Urban_Share', 'Rule_of_Law'],
    'Urban only': ['Urban_Share']
}

# === Loop through model specs ===
for name, vars in models.items():
    print(f"\n\n=== {name.upper()} ===")

    X = df[vars]
    X_ols = sm.add_constant(X)

    # --- Pooled OLS
    pooled_model = sm.OLS(y, X_ols).fit()
    print("\n--- Pooled OLS ---")
    print(pooled_model.summary())

    # --- Fixed Effects
    fe_model = PanelOLS(y, X, entity_effects=True)
    fe_results = fe_model.fit(cov_type='robust')
    print("\n--- Fixed Effects (Robust SEs) ---")
    print(fe_results.summary)

    # --- VIF
    if len(vars) > 1:  # Only if more than one predictor
        print("\n--- VIF ---")
        vif_df = pd.DataFrame()
        vif_df["Variable"] = X.columns
        vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print(vif_df)

    # --- Breusch-Pagan (OLS residuals)
    bp_test = het_breuschpagan(pooled_model.resid, X_ols)
    bp_labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    print("\n--- Breusch-Pagan Test ---")
    print(dict(zip(bp_labels, bp_test)))

# === Optional: Correlation heatmap (from full model) ===
print("\n=== Correlation Heatmap (Full Variables) ===")
sns.heatmap(df[['Urban_Share', 'Rule_of_Law', 'Governance_Index']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
