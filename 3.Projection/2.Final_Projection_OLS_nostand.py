# ================================================================
# 0. Imports
# ================================================================
import pandas as pd

# ---------------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------------
wacc_path = (
    "/Users/valentinadlc/Documents/MASTER/MASTER THESIS/WACC_Thesis_DLC/1.Cleaning_and_Merges/final_wacc_macro_historical.csv"
)
macro_path = (
    "/Users/valentinadlc/Documents/MASTER/MASTER THESIS/WACC_Thesis_DLC/1.Cleaning_and_Merges/ngfs_final_merge.csv"
)

# ---------------------------------------------------------------
# 2. Load data and clean units
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
# 3. Add technology dummies to historical WACC data
# ---------------------------------------------------------------
wacc_df["is_solar"] = wacc_df["technology"].str.contains("solar", case=False, na=False).astype(int)
wacc_df["is_wind_offshore"] = wacc_df["technology"].str.contains("offshore", case=False, na=False).astype(int)

# ---------------------------------------------------------------
# 4. Pooled-OLS coefficients (from regression on raw data)
# ---------------------------------------------------------------
coef = {
    "const": 0.0683,
    "gdp_ppp": -0.000002317,         # or -2.317e-06
    "population": 0.0000000000279,   # or 2.79e-11
    "inflation": 0.0027,
    "unemployment": 0.0016,
    "is_solar": 0.0200,
    "is_wind_offshore": 0.0076,
}

# ---------------------------------------------------------------
# 5. Generate projections by technology type
# ---------------------------------------------------------------
tech_map = {
    "Wind_Onshore": {"is_solar": 0, "is_wind_offshore": 0},
    "Solar_PV":     {"is_solar": 1, "is_wind_offshore": 0},
    "Wind_Offshore": {"is_solar": 0, "is_wind_offshore": 1},
}

proj_frames = []
for tech, dums in tech_map.items():
    tmp = macro_df.copy()
    tmp["Technology"] = tech
    tmp["is_solar"] = dums["is_solar"]
    tmp["is_wind_offshore"] = dums["is_wind_offshore"]

    tmp["wacc_projection"] = (
        coef["const"]
        + coef["gdp_ppp"] * tmp["gdp_ppp"]
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
    "/Users/valentinadlc/Documents/MASTER/MASTER THESIS/WACC_Thesis_DLC/3.Projection/wacc_projection_by_scenario.csv"
)
proj_df.to_csv(out_path, index=False)

print("\n✅ Projections written to:", out_path)
print(proj_df.head())
