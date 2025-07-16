# === Importing libraries ===
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np

# === Ignore common warnings ===
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# === Load and prepare dataset ===
merged = pd.read_csv("/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/merged_with_wacc_updated.csv")
merged.drop(columns=['interest_rate'], inplace=True, errors='ignore')  # Avoid error if already dropped

# Create technology dummies
merged['is_solar'] = merged['technology'].str.contains('solar', case=False, na=False).astype(int)
merged['is_wind_onshore'] = merged['technology'].str.contains('onshore', case=False, na=False).astype(int)
merged['is_wind_offshore'] = merged['technology'].str.contains('offshore', case=False, na=False).astype(int)

# Drop rows with missing data for selected columns
df = merged.dropna(subset=[
    'wacc', 'population', 'gdp', 'inflation', 'unemployment',
    'is_solar', 'is_wind_onshore', 'is_wind_offshore'
]).copy()

# Standardize selected features
features = ['population', 'gdp', 'inflation', 'unemployment']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Set index (only used for organization here, not needed for OLS)
df.set_index(['country', 'year'], inplace=True)

# === EDA: Correlation Matrix ===
plt.figure(figsize=(8, 6))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Predictors")
plt.show()

# === EDA: Distributions ===
df[features + ['wacc']].hist(bins=20, figsize=(12, 8), layout=(2, 3))
plt.suptitle("Distributions of Key Variables", y=1.02)
plt.tight_layout()
plt.show()

# === EDA: Scatterplots ===
for col in features:
    sns.scatterplot(data=df, x=col, y='wacc')
    plt.title(f'WACC vs {col}')
    plt.grid(True)
    plt.show()

# === Multicollinearity test (VIF) ===
X_vif = sm.add_constant(df[[
    'population', 'gdp', 'inflation', 'unemployment',
    'is_wind_onshore', 'is_wind_offshore'
]])
vif_data = pd.DataFrame({
    "Variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print("\n=== Variance Inflation Factor (VIF) ===")
print(vif_data)

# === Simple OLS Regression (No Entity Effects) ===
df_reset = df.reset_index()
X = df_reset[[
    'population', 'gdp', 'inflation', 'unemployment',
    'is_solar', 'is_wind_onshore', 'is_wind_offshore'
]]
X = sm.add_constant(X)
y = df_reset['wacc']

ols_model = sm.OLS(y, X).fit()

print("\n=== OLS Regression (No Entity Effects) ===")
print(ols_model.summary())

# === Extract intercept and coefficients ===
intercept = ols_model.params['const']
coefficients = ols_model.params.drop('const')

print("\n📌 Intercept for projection:", intercept)
print("📌 Coefficients for projection:")
print(coefficients)

# === Save coefficients if needed ===
coefficients.to_csv("wacc_projection_coefficients.csv")

# === Save full merged dataset (optional backup) ===
df_reset.to_csv("wacc_data_for_projection.csv", index=False)
