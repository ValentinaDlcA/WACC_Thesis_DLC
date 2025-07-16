# ================================================================
# 0. Imports
# ================================================================
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------
# 1. File paths  – adjust only if your directory changes
# ---------------------------------------------------------------
wacc_path  = (
    "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/"
    "demo/PyCharmLearningProject/data1/wacc_with_gdpppp2021.csv"
)
macro_path = (
    "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/"
    "demo/PyCharmLearningProject/WACC Thesis/macro_gdp_merged.csv"
)

# ---------------------------------------------------------------
# 2. Load data and align column names
# ---------------------------------------------------------------
wacc_df  = pd.read_csv(wacc_path)
macro_df = pd.read_csv(macro_path)

# ── make sure the PPP column has the SAME name in both frames ──
wacc_df  = wacc_df.rename(columns={"GDP_PPP": "gdp_ppp"})
macro_df = macro_df.rename(columns={"GDP_PPP": "gdp_ppp"})   # in case

# ▸ diagnostic print
print("WACC columns :", wacc_df.columns.tolist())
print("Macro columns:", macro_df.columns.tolist())

# ---------------------------------------------------------------
# 3. Technology dummies in the historical WACC file
#    (needed for the scaler fit)
# ---------------------------------------------------------------
wacc_df["is_solar"]         = wacc_df["technology"].str.contains("solar",    case=False, na=False).astype(int)
wacc_df["is_wind_offshore"] = wacc_df["technology"].str.contains("offshore", case=False, na=False).astype(int)

# ---------------------------------------------------------------
# 4. Build and store the scaler on historical data
# ---------------------------------------------------------------
cont_vars = ["gdp_ppp", "inflation", "unemployment"]
scaler = StandardScaler().fit(wacc_df[cont_vars])

print("\nScaler means :", dict(zip(cont_vars, scaler.mean_)))
print("Scaler stddev:", dict(zip(cont_vars, scaler.scale_)))

# ---------------------------------------------------------------
# 5. Apply the same scaling to macro-scenario inputs
# ---------------------------------------------------------------
macro_std = macro_df.copy()
macro_std[cont_vars] = scaler.transform(macro_std[cont_vars])

# ---------------------------------------------------------------
# 6. Pooled-OLS coefficients (population term omitted)
#    Reference technology = Wind-On-shore
# ---------------------------------------------------------------
coef = {
    "const"            : 0.0683,
    "gdp_ppp"          : -0.0127,
    "inflation"        :  0.0203,
    "unemployment"     :  0.0089,
    "is_solar"         :  0.0200,
    "is_wind_offshore" :  0.0076,
}

# ---------------------------------------------------------------
# 7. Build projections for each technology
# ---------------------------------------------------------------
tech_map = {
    "Wind_Onshore" : {"is_solar": 0, "is_wind_offshore": 0},
    "Solar_PV"     : {"is_solar": 1, "is_wind_offshore": 0},
    "Wind_Offshore": {"is_solar": 0, "is_wind_offshore": 1},
}

proj_frames = []
for tech, dums in tech_map.items():
    tmp = macro_std.copy()
    tmp["Technology"]        = tech
    tmp["is_solar"]          = dums["is_solar"]
    tmp["is_wind_offshore"]  = dums["is_wind_offshore"]

    tmp["wacc_projection"] = (
          coef["const"]
        + coef["gdp_ppp"]         * tmp["gdp_ppp"]
        + coef["inflation"]       * tmp["inflation"]
        + coef["unemployment"]    * tmp["unemployment"]
        + coef["is_solar"]        * tmp["is_solar"]
        + coef["is_wind_offshore"]* tmp["is_wind_offshore"]
    )
    proj_frames.append(
        tmp[["Scenario", "ISO", "Year", "Technology", "wacc_projection"]]
    )

proj_df = pd.concat(proj_frames, ignore_index=True)

# ---------------------------------------------------------------
# 8. Save result
# ---------------------------------------------------------------
out_path = (
    "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/"
    "demo/PyCharmLearningProject/WACC Thesis/wacc_projection_by_scenario.csv"
)
proj_df.to_csv(out_path, index=False)
print("\n✅ Projections written to:", out_path)
print(proj_df.head())


