# 1. GDP PPP CLEANING
import pandas as pd
import pycountry

# === Load the raw GDP PPP data ===
gdp_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data/gdp_ppp.csv"  # Update this with your actual path
gdp_df = pd.read_csv(gdp_path)

# === Drop unnecessary columns ===
gdp_df = gdp_df.drop(columns=["Model", "Unit"])

# === Rename scenarios ===
scenario_name_map = {
    "Below 2?C": "Below2",
    "Nationally Determined Contributions (NDCs)": "NDC",
    "Net Zero 2050": "Netzero"
}
gdp_df["Scenario"] = gdp_df["Scenario"].replace(scenario_name_map)

# === Remove "Current Policies" scenario ===
gdp_df = gdp_df[gdp_df["Scenario"] != "Current Policies"]

# === Standardize Variable column ===
gdp_df["Variable"] = "gdp_ppp"

# === Rename Region to ISO ===
gdp_df = gdp_df.rename(columns={"Region": "ISO"})

# === Add Country column based on ISO ===
def iso_to_country(iso_code):
    try:
        return pycountry.countries.get(alpha_3=iso_code).name
    except:
        return None

gdp_df["Country"] = gdp_df["ISO"].apply(iso_to_country)

# === Reorder columns to place Country next to ISO ===
cols = gdp_df.columns.tolist()
if "ISO" in cols and "Country" in cols:
    iso_index = cols.index("ISO")
    cols.insert(iso_index + 1, cols.pop(cols.index("Country")))
    gdp_df = gdp_df[cols]

# === Convert to long format: one row per year ===
id_vars = ["Scenario", "ISO", "Country", "Variable"]
value_vars = [col for col in gdp_df.columns if col not in id_vars]

gdp_ppp_long = gdp_df.melt(
    id_vars=id_vars,
    value_vars=value_vars,
    var_name="Year",
    value_name="gdp_ppp"
)

# Preview
print(gdp_ppp_long.head())

#CLEAN WACC FILE
# === Load the WACC file ===
wacc_raw = pd.read_excel("/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data/WACC.xlsx")

# === Manual ISO patch for unrecognized countries ===
manual_iso_map = {
    "Taiwan, China": "TWN"
}

def get_iso_alpha3_patched(country):
    if country in manual_iso_map:
        return manual_iso_map[country]
    try:
        return pycountry.countries.lookup(country).alpha_3
    except:
        return None

# === Step 1: Keep only necessary columns ===
wacc_cleaned = wacc_raw[["Country", "Technology", "Year", "WACC AT"]].copy()
wacc_cleaned = wacc_cleaned.rename(columns={
    "Technology": "technology",
    "WACC AT": "wacc"
})

# === Step 2: Add technology dummy variables ===
wacc_cleaned["is_solar"] = (wacc_cleaned["technology"] == "Solar PV").astype(int)
wacc_cleaned["is_wind_onshore"] = (wacc_cleaned["technology"] == "Wind onshore").astype(int)
wacc_cleaned["is_wind_offshore"] = (wacc_cleaned["technology"] == "Wind offshore").astype(int)

# === Step 3: Add ISO codes from Country (patched) ===
wacc_cleaned["ISO"] = wacc_cleaned["Country"].apply(get_iso_alpha3_patched)
wacc_cleaned = wacc_cleaned[wacc_cleaned["ISO"].notna()]

# === Step 4: Add 'Scenario' and 'Variable' columns to match GDP PPP format ===
wacc_cleaned["Scenario"] = None
wacc_cleaned["Variable"] = "wacc"

# === Step 5: Reorder columns (no 'Scenario') ===
wacc_cleaned = wacc_cleaned[[
    "ISO", "Country", "Variable", "Year", "wacc",
    "is_solar", "is_wind_onshore", "is_wind_offshore"
]]

# === Step 6: Save or preview ===
# wacc_cleaned.to_csv("wacc_cleaned.csv", index=False)
print(wacc_cleaned.head())
print("✅ WACC cleaned. Total unique countries:", wacc_cleaned['ISO'].nunique())

wacc_cleaned.to_csv("wacc_cleaned.csv", index=False)

#MACROFINAL CLEANING
# === Load macrofinal ===
macro_raw = pd.read_csv("/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data1/macrofinal.csv")  # Update with your path

# 1. Remove 'Model' column
if 'Model' in macro_raw.columns:
    macro_raw = macro_raw.drop(columns=["Model"])

# 2. Extract country names from 'Region'
macro_raw["Country"] = macro_raw["Region"].str.replace(r"^NiGEM NGFS v[\d\.]+\|\s*", "", regex=True)

# 3. Remove aggregate regions
regions_to_exclude = [
    "Africa", "Asia", "Central America", "Europe", "Middle East", "North America",
    "South America", "World", "Rest of the World", "European Union", "Pacific Island States"
]
macro_raw = macro_raw[~macro_raw["Country"].isin(regions_to_exclude)]

# 4. Rename scenarios
scenario_name_map = {
    "Below 2?C": "Below2",
    "Nationally Determined Contributions (NDCs)": "NDC",
    "Net Zero 2050": "Netzero"
}
macro_raw["Scenario"] = macro_raw["Scenario"].replace(scenario_name_map)

# 5. Keep only inflation and unemployment
keep_vars = [
    "Unemployment rate ; %",
    "Inflation rate ; %",
    "Unemployment rate ; %(combined)",
    "Inflation rate ; %(combined)"
]
macro_raw = macro_raw[macro_raw["Variable"].isin(keep_vars)]

# 6. Normalize variable names
variable_rename_map = {
    "Unemployment rate ; %": "unemployment",
    "Unemployment rate ; %(combined)": "unemployment",
    "Inflation rate ; %": "inflation",
    "Inflation rate ; %(combined)": "inflation"
}
macro_raw["Variable"] = macro_raw["Variable"].map(variable_rename_map)

# 7. Add ISO codes
def get_iso(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

macro_raw["ISO"] = macro_raw["Country"].apply(get_iso)

# Manually patch ISO if needed
manual_iso_map = {
    "Russia": "RUS",
    "South Korea": "KOR"
}
macro_raw["ISO"] = macro_raw.apply(lambda row: manual_iso_map.get(row["Country"], row["ISO"]), axis=1)
macro_raw = macro_raw[macro_raw["ISO"].notna()]

# 8. Reorder columns
year_cols = [col for col in macro_raw.columns if col.isdigit()]
macro_cleaned = macro_raw[["Scenario", "ISO", "Country", "Variable"] + year_cols]

# 9. Melt to long format
macro_cleaned = macro_cleaned.melt(
    id_vars=["Scenario", "ISO", "Country", "Variable"],
    var_name="Year",
    value_name="Value"
)

# Rename after melting (optional)
# If this is the first time you're melting:
macro_long = pd.melt(
    macro_cleaned,
    id_vars=["Scenario", "ISO", "Country", "Variable"],
    var_name="Year",
    value_name="MacroValue"
)

#Rename to 'Value' if needed later
macro_long = macro_long.rename(columns={"MacroValue": "Value"})

# Ensure Year is string
macro_long["Year"] = macro_long["Year"].astype(str)

print(macro_long.head())

# 10. Preview
print(macro_long.head())
print("✅ Macro cleaned. Rows:", len(macro_long), " | Countries:", macro_long['ISO'].nunique())
print("🔍 Columns in macro_cleaned:", macro_cleaned.columns.tolist())

#CALCULATING SCENARIO VALUES FROM BASELINE IN MACRO_LONG
# 1. Separate baseline and non-baseline data
baseline_df = macro_cleaned[macro_cleaned["Scenario"] == "Baseline"]
scenarios = macro_cleaned["Scenario"].unique().tolist()
scenarios = [s for s in scenarios if s != "Baseline"]



# 2. Merge baseline with each scenario by ISO, Variable, and Year
reconstructed_frames = []

for scenario in scenarios:
    scenario_df = macro_cleaned[macro_cleaned["Scenario"] == scenario]

    merged = pd.merge(
        scenario_df,
        baseline_df,
        on=["ISO", "Variable", "Year"],
        suffixes=("_scenario", "_baseline"),
        how="inner"
    )

    # 3. Reconstruct actual values by adding delta to baseline
    merged["Value"] = merged["Value_baseline"] + merged["Value_scenario"]

    # 4. Build cleaned output
    merged_cleaned = merged[["Scenario_scenario", "ISO", "Variable", "Year", "Value"]].rename(
        columns={"Scenario_scenario": "Scenario"}
    )

    # Append to results
    reconstructed_frames.append(merged_cleaned)

# 5. Add original baseline data back in
baseline_cleaned = baseline_df[["Scenario", "ISO", "Variable", "Year", "Value"]]

# 6. Concatenate all into macro_reconstructed
macro_reconstructed = pd.concat([baseline_cleaned] + reconstructed_frames, ignore_index=True)

# Optional: Add back country info if you need it
macro_reconstructed = pd.merge(
    macro_reconstructed,
    macro_cleaned[["ISO", "Country"]].drop_duplicates(),
    on="ISO",
    how="left"
)

# ✅ Preview result
print(macro_reconstructed.head())

#MERGE GDP_PPP AND MACRO CLEANED
# Step 1: Pivot macro_reconstructed to wide format
macro_reconstructed_wide = macro_reconstructed.pivot_table(
    index=["Scenario", "ISO", "Country", "Year"],
    columns="Variable",
    values="Value"
).reset_index()

# Define forecast years (every 5 years from 2025 to 2050)
forecast_years = [str(y) for y in range(2025, 2051, 5)]

# Step 2:Filter GDP PPP for forecast years
gdp_forecast = gdp_ppp_long[
    gdp_ppp_long["Year"].isin(forecast_years)
][["ISO", "Scenario", "Year", "gdp_ppp"]]

# Step 3: Merge with gdp_forecast
macro_gdp_merged = pd.merge(
    macro_reconstructed_wide,
    gdp_forecast,
    on=["ISO", "Scenario", "Year"],
    how="inner"
)

# ✅ Preview result
print(macro_gdp_merged.head())

# Save reshaped macro data to file
macro_long.to_csv("macro_long_cleaned.csv", index=False)

# Save merged macro + GDP PPP data
macro_gdp_merged.to_csv("macro_gdp_merged.csv", index=False)
print("✅ Saved: macro_gdp_merged.csv")

#HOW MANY COUNTRIES PER SCENARIO AFTER MERGE = IN ALL 48 COUNTRIES
country_counts = macro_gdp_merged.groupby("Scenario")["ISO"].nunique().reset_index()
country_counts.columns = ["Scenario", "Num_Countries"]

print(country_counts)