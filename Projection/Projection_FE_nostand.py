# ================================================================
# 0. Imports
# ================================================================
import pandas as pd

# ---------------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------------
wacc_path = (
    "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/"
    "PyCharmLearningProject/WACC Thesis/final_wacc_macro_historical.csv"
)
macro_path = (
    "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/"
    "PyCharmLearningProject/WACC Thesis/ngfs_final_merge.csv"
)

# ---------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------
wacc_df = pd.read_csv(wacc_path)
macro_df = pd.read_csv(macro_path)

# Convert NGFS population from millions to total
macro_df['population'] = macro_df['population'] * 1_000_000

# Rename GDP_PPP column for consistency
wacc_df = wacc_df.rename(columns={"GDP_PPP": "gdp_ppp"})
macro_df = macro_df.rename(columns={"GDP_PPP": "gdp_ppp"})

# Diagnostic print
print("WACC columns :", wacc_df.columns.tolist())
print("Macro columns:", macro_df.columns.tolist())

# ---------------------------------------------------------------
# 3. Add technology dummies to macro data
# ---------------------------------------------------------------
tech_map = {
    "Wind_Onshore": {"is_solar": 0, "is_wind_offshore": 0},
    "Solar_PV":     {"is_solar": 1, "is_wind_offshore": 0},
    "Wind_Offshore": {"is_solar": 0, "is_wind_offshore": 1},
}

# ---------------------------------------------------------------
# 4. Fixed Effects model coefficients (from regression)
# ---------------------------------------------------------------
coef = {
    # No constant in FE
    "gdp_ppp": -0.0274,
    "population": 0.0682,
    "inflation": 0.0179,
    "unemployment": 0.0209,
    "is_solar": 0.0006,
    "is_wind_offshore": 0.0157,
}

# ---------------------------------------------------------------
# 5. Generate projections by technology type
# ---------------------------------------------------------------
proj_frames = []
for tech, dums in tech_map.items():
    tmp = macro_df.copy()
    tmp["Technology"] = tech
    tmp["is_solar"] = dums["is_solar"]
    tmp["is_wind_offshore"] = dums["is_wind_offshore"]

    tmp["wacc_projection"] = (
        coef["gdp_ppp"] * tmp["gdp_ppp"]
        + coef["population"] * tmp["population"]
        + coef["inflation"] * tmp["inflation"]
        + coef["unemployment"] * tmp["unemployment"]
        + coef["is_solar"] * tmp["is_solar"]
        + coef["is_wind_offshore"] * tmp["is_wind_offshore"]
    )

    proj_frames.append(
        tmp[["Scenario", "ISO", "Year", "Technology", "wacc_projection"]]
    )

proj_df = pd.concat(proj_frames, ignore_index=True)

# ---------------------------------------------------------------
# 6. Save final projection result
# ---------------------------------------------------------------
out_path = (
    "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/"
    "demo/PyCharmLearningProject/WACC Thesis/wacc_projection_FE.csv"
)
proj_df.to_csv(out_path, index=False)

print("\n✅ Fixed Effects Projections written to:", out_path)
print(proj_df.head())
