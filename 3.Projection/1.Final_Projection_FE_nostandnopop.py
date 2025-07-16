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
# 2. Load data and convert units
# ---------------------------------------------------------------
wacc_df = pd.read_csv(wacc_path)
macro_df = pd.read_csv(macro_path)

# Convert NGFS population from millions to total
macro_df['population'] = macro_df['population'] * 1_000_000

# Rename GDP_PPP for consistency
wacc_df = wacc_df.rename(columns={"GDP_PPP": "gdp_ppp"})
macro_df = macro_df.rename(columns={"GDP_PPP": "gdp_ppp"})

print("WACC columns :", wacc_df.columns.tolist())
print("Macro columns:", macro_df.columns.tolist())

# ---------------------------------------------------------------
# 3. FE coefficients (no constant, no population)
# ---------------------------------------------------------------
coef = {
    "gdp_ppp": -2.943e-06,
    "inflation": 0.0024,
    "unemployment": 0.0038,
    "is_solar": 0.0005,
    "is_wind_offshore": 0.0154,
}

# ---------------------------------------------------------------
# 4. Technology dummy map
# ---------------------------------------------------------------
tech_map = {
    "Wind_Onshore": {"is_solar": 0, "is_wind_offshore": 0},
    "Solar_PV":     {"is_solar": 1, "is_wind_offshore": 0},
    "Wind_Offshore": {"is_solar": 0, "is_wind_offshore": 1},
}

# ---------------------------------------------------------------
# 5. Generate projections
# ---------------------------------------------------------------
proj_frames = []

for tech, dummies in tech_map.items():
    tmp = macro_df.copy()
    tmp["Technology"] = tech
    tmp["is_solar"] = dummies["is_solar"]
    tmp["is_wind_offshore"] = dummies["is_wind_offshore"]

    tmp["wacc_projection"] = (
        coef["gdp_ppp"] * tmp["gdp_ppp"]
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
# 6. Save final output
# ---------------------------------------------------------------
out_path = (
    "/Users/valentinadlc/Documents/MASTER/MASTER THESIS/WACC_Thesis_DLC/3.Projection/wacc_projection_FE_nopop.csv"
)
proj_df.to_csv(out_path, index=False)

print("\n✅ FE projection (without population) written to:")
print(out_path)
print(proj_df.head())
