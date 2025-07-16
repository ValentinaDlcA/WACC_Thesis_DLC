import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np

# Ignore SettingWithCopyWarning and divide by zero warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# === Load and prepare dataset ===
merged = pd.read_csv("../WACC_Thesis_old/merged_with_wacc.csv")
merged.drop(columns=['interest_rate'], inplace=True)

# Create dummies
merged['is_solar'] = merged['technology'].str.contains('solar', case=False).astype(int)
merged['is_wind_onshore'] = merged['technology'].str.contains('onshore', case=False).astype(int)
merged['is_wind_offshore'] = merged['technology'].str.contains('offshore', case=False).astype(int)

# Drop rows with missing values
df = merged.dropna(subset=[
    'wacc', 'population', 'gdp', 'inflation', 'unemployment',
    'is_solar', 'is_wind_onshore', 'is_wind_offshore'
]).copy()

# Standardize independent variables
scaler = StandardScaler()
features = ['population', 'gdp', 'inflation', 'unemployment']
df.loc[:, features] = scaler.fit_transform(df[features])

# Set multi-index for panel data
df = df.set_index(['country', 'year'])

# === EDA: Correlation matrix ===
plt.figure(figsize=(8, 6))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Predictors")
plt.show()

# === EDA: Distributions ===
df[features + ['wacc']].hist(bins=20, figsize=(12, 8), layout=(2, 3))
plt.suptitle("Distributions of Key Variables", y=1.02)
plt.tight_layout()
plt.show()

# === EDA: Scatterplots (WACC vs predictors) ===
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

vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print("\n=== Variance Inflation Factor (VIF) ===")
print(vif_data)

# === Regression with fixed effects (PanelOLS) ===
X = df[[
    'population', 'gdp', 'inflation', 'unemployment',
    'is_solar', 'is_wind_onshore', 'is_wind_offshore'
]]
y = df['wacc']

model = PanelOLS(y, X, entity_effects=True)
results = model.fit()

print("\n=== Fixed Effects Regression Results ===")
print(results.summary)


# === Residual diagnostics ===
residuals = results.resids.squeeze()  # Convert to 1D array
fitted = results.fitted_values.squeeze()  # Convert to 1D array

# Residual vs Fitted
sns.scatterplot(x=fitted, y=residuals)
plt.axhline(0, linestyle='--', color='black')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.grid(True)
plt.show()

# QQ plot for normality of residuals
sm.qqplot(residuals, line='45', fit=True)
plt.title("QQ Plot of Residuals")
plt.show()

# === Outlier and influence analysis ===
# For outlier detection, re-fit with OLS on flattened data (PanelOLS doesn't support influence)
df_reset = df.reset_index()
X_ols = sm.add_constant(df_reset[[
    'population', 'gdp', 'inflation', 'unemployment',
    'is_wind_onshore', 'is_wind_offshore'
]])
y_ols = df_reset['wacc']
ols_model = sm.OLS(y_ols, X_ols).fit()
influence = OLSInfluence(ols_model)

# Cook's distance
plt.stem(np.arange(len(influence.cooks_distance[0])), influence.cooks_distance[0], markerfmt=",")
plt.title("Cook's Distance for Outlier Detection")
plt.xlabel("Observation Index")
plt.ylabel("Cook's Distance")
plt.grid(True)
plt.show()

# Leverage plot
sns.scatterplot(x=influence.hat_matrix_diag, y=influence.resid_studentized_internal)
plt.xlabel("Leverage")
plt.ylabel("Studentized Residuals")
plt.title("Leverage vs Residuals")
plt.axhline(0, linestyle='--', color='black')
plt.grid(True)
plt.show()

# === Summary stats ===
print("\nNumber of countries:", merged['country'].nunique())

# === Plot: WACC over time by technology ===
plt.figure(figsize=(12, 8))
for tech, color in zip(['is_solar', 'is_wind_onshore', 'is_wind_offshore'],
                       ['red', 'blue', 'green']):
    tech_data = merged[merged[tech] == 1]
    plt.scatter(tech_data['year'], tech_data['wacc'], label=tech.split('_')[-1].capitalize(), color=color, alpha=0.5)

plt.xlabel('Year')
plt.ylabel('WACC')
plt.title('WACC over Years by Technology')
plt.legend()
plt.grid(True)
plt.show()

# === Plot: WACC by country and technology (2023) ===
plt.figure(figsize=(14, 8))
data_2023 = merged[merged['year'] == 2023]

for tech, color in zip(['is_solar', 'is_wind_onshore', 'is_wind_offshore'],
                       ['red', 'blue', 'green']):
    tech_data = data_2023[data_2023[tech] == 1]
    plt.bar(tech_data['country'], tech_data['wacc'], label=tech.split('_')[-1].capitalize(), alpha=0.5, color=color)

plt.xticks(rotation=45, ha='right')
plt.xlabel('Country')
plt.ylabel('WACC')
plt.title('WACC by Country and Technology (2023)')
plt.legend()
plt.tight_layout()
plt.show()
# === Save updated file ===
merged.to_csv("merged_with_wacc_updated.csv", index=False)

