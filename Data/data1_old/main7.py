import pandas as pd

# === Step 1: Load datasets ===
proj_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data1/merged_macro_popgdp.csv"
hist_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/merged_with_wacc_wind_inflation.csv"

df_proj = pd.read_csv(proj_path)
df_hist = pd.read_csv(hist_path)

# === Step 2: Rename GDP and population columns in projections ===
df_proj = df_proj.rename(columns={
    "GDP_damage": "gdp",
    "Population": "population"
})

# === Step 3: Add missing columns for regression formula ===
for col in ["is_wind", "wind_inflation"]:
    if col not in df_proj.columns:
        df_proj[col] = 0

# === Step 4: Fill missing gdp/population with historical means ===
if "population" in df_hist.columns:
    df_proj["population"] = df_proj["population"].fillna(df_hist["population"].mean())
if "gdp" in df_hist.columns:
    df_proj["gdp"] = df_proj["gdp"].fillna(df_hist["gdp"].mean())

# === Step 5: Apply regression formula to project WACC ===
df_proj["WACC_projected"] = (
    0.0874 +
    0.0049 * df_proj["population"] +
    -0.0090 * df_proj["gdp"] +
    0.0219 * df_proj["inflation"] +
    0.0067 * df_proj["unemployment"] +
    -0.0203 * df_proj["is_wind"] +
)

# === Step 6: Save to CSV ===
output_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data1/wacc_projected_scenarios.csv"
df_proj.to_csv(output_path, index=False)

# === Step 7: Preview output ===
print(df_proj[["Scenario", "Country", "Year", "inflation", "unemployment", "WACC_projected"]].head())
