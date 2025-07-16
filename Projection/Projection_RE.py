# ================================================================
# 0. Imports
# ================================================================
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
# 2. Load data and standardize units
# ---------------------------------------------------------------
wacc_df = pd.read_csv(wacc_path)
macro_df = pd.read_csv(macro_path)

# Convert NGFS population from millions to total
macro_df['population'] = macro_df['population'] * 1_000_000

# Rename GDP_PPP column for consistency
wacc_df = wacc_df.rename(columns={"GDP_PPP": "gdp_ppp"})
macro_df = macro_df.rename(columns={"GDP_PPP": "gdp_ppp"})

# ---------------------------------------------------------------
# 3. Add technology dummies to historical WACC data
# ---------------------------------------------------------------
wacc_df["is_solar"] = wacc_df["technology"].str.contains("solar", case=False, na=False).astype(int)
wacc_df["is_wind_offshore"] = wacc_df["technology"].str.contains("offshore", case=False, na=False).astype(int)

# ---------------------------------------------------------------
# 4. Fit scaler on macroeconomic variables (including population)
# ---------------------------------------------------------------
cont_vars = ["gdp_ppp", "inflation", "unemployment", "population"]
scaler = StandardScaler().fit(wacc_df[cont_vars])

# Apply scaling to macro-scenario data
macro_std = macro_df.copy()
macro_std[cont_vars] = scaler.transform(macro_std[cont_vars])

# ---------------------------------------------------------------
# 5. Random Effects model coefficients
# ---------------------------------------------------------------
coef = {
    "gdp_ppp": -0.0286,
    "population": 0.0164,
    "inflation": 0.0162,
    "unemployment": 0.0115,
    "is_solar": 0.0035,
    "is_wind_offshore": 0.0215,
}

# ---------------------------------------------------------------
# 6. Generate projections by technology type
# ---------------------------------------------------------------
tech_map = {
    "Wind_Onshore": {"is_solar": 0, "is_wind_offshore": 0},
    "Solar_PV":     {"is_solar": 1, "is_wind_offshore": 0},
    "Wind_Offshore": {"is_solar": 0, "is_wind_offshore": 1},
}

proj_frames = []
for tech, dums in tech_map.items():
    tmp = macro_std.copy()
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
# 7. Save final projection result
# ---------------------------------------------------------------
out_path = (
    "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/"
    "demo/PyCharmLearningProject/WACC Thesis/wacc_projection_RE.csv"
)
proj_df.to_csv(out_path, index=False)

print("\n✅ Random Effects Projections written to:", out_path)
print(proj_df.head())
