import pandas as pd
import numpy as np
import pycountry
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# === Helper Function to Convert Country Names to ISO Codes ===
def get_iso_alpha3(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

# 1. File Paths
macro_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data1/macrofinal.csv"
wacc_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/merged_with_wacc_updated.csv"
gdp_cleaned_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data/gdp_ppp_cleaned.csv"

macro_df = pd.read_csv(macro_path)
wacc_df = pd.read_csv(wacc_path)
gdp_cleaned_df = pd.read_csv(gdp_cleaned_path)

# 2. Cleaning Macrofinal
macro_df["Country"] = macro_df["Region"].str.replace(r"^NiGEM NGFS v1\.24\.2\|\s*", "", regex=True)

regions_to_exclude = [
    "Africa", "Asia", "Central America", "Europe", "Middle East", "North America",
    "South America", "World", "Rest of the World", "European Union", "Pacific Island States"
]
macro_df = macro_df[~macro_df["Country"].isin(regions_to_exclude)].copy()

# Define the mapping dictionary for scenario name changes
scenario_name_map = {
    "Below 2?C": "Below2",
    "Nationally Determined Contributions (NDCs)": "NDC",
    "Net Zero 2050": "Netzero"
}

# Apply the mapping to the 'Scenario' column
macro_df['Scenario'] = macro_df['Scenario'].replace(scenario_name_map)

# Count unique scenarios in the original macrofinal DataFrame
unique_scenarios_macro = macro_df['Scenario'].nunique()
unique_scenarios_list_macro = macro_df['Scenario'].unique().tolist()
print(f"Number of unique scenarios in macrofinal.csv: {unique_scenarios_macro}")
print(f"Scenarios in macrofinal.csv: {unique_scenarios_list_macro}")

# Remove rows with 'Gross' in the 'Variable' column (just in case any slipped through)
macro_df = macro_df[~macro_df["Variable"].str.contains("Gross", na=False)]

# Normalize variable names
variable_rename_map = {
    "Unemployment rate ; %": "unemployment",
    "Inflation rate ; %": "inflation",
    "Unemployment rate ; %(combined)": "unemployment",
    "Inflation rate ; %(combined)": "inflation",
}
macro_df["Variable"] = macro_df["Variable"].map(variable_rename_map)
macro_df = macro_df[macro_df["Variable"].notna()].copy()

# ISO Conversion
macro_df["ISO"] = macro_df["Country"].apply(get_iso_alpha3)
manual_iso_map = {
    "Russia": "RUS",
    "South Korea": "KOR"
}
macro_df["ISO"] = macro_df.apply(lambda row: manual_iso_map.get(row["Country"], row["ISO"]), axis=1)
macro_df = macro_df[macro_df["ISO"].notna()].copy()

print("Updated gdp_cleaned_df columns:", gdp_cleaned_df.columns.tolist())

# 3. Merge (macro, wacc, gdp_ppp) data frames

# Merge macro and wacc on ISO and Scenario (if present in both)
merged_temp = pd.merge(
    macro_df,
    wacc_df,
    on=["ISO"],
    suffixes=("_macro", "_wacc")
)

# Then merge with gdp_ppp
merged_macro_wacc_gdp = pd.merge(
    merged_temp,
    gdp_cleaned_df,
    on=["ISO"],
    suffixes=("", "_gdp_cleaned")
)

print(merged_macro_wacc_gdp.head())

# 5. Train Regression Model
features = ["inflation", "unemployment", "gdp_ppp", "is_solar", "is_wind_onshore", "is_wind_offshore"]
X = merged_macro_wacc_gdp[features]
y = merged_macro_wacc_gdp["wacc"]
X = X.dropna()
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# === Rebuild Scenarios ===
baseline_df = macro_df[macro_df["Scenario"] == "Baseline"].copy()
scenario_names = macro_df["Scenario"].dropna().unique()
scenario_names = [s for s in scenario_names if s != "Baseline"]
years = [str(y) for y in range(2022, 2051)]

scenario_dfs = {}
for scenario in scenario_names:
    scenario_df = macro_df[macro_df["Scenario"] == scenario].copy()
    merged = pd.merge(
        scenario_df,
        baseline_df,
        on=["Country", "Variable", "ISO"],
        suffixes=("_diff", "_base")
    )
    for year in years:
        merged[year] = merged[f"{year}_base"] + merged[f"{year}_diff"]
    final_cols = ["Scenario_diff", "Country", "Variable", "ISO"] + years
    scenario_dfs[scenario] = merged[final_cols].rename(columns={"Scenario_diff": "Scenario"})

#Changing scenarios name
for scenario in scenario_dfs:
    scenario_dfs[scenario]['Scenario'] = scenario_dfs[scenario]['Scenario'].replace(scenario_name_map)

# === Forecast Function ===
def forecast_wacc(model, scenario_df, years, tech_name, tech_dummies):
    forecasts = []
    countries = scenario_df['Country'].unique()

    for country in countries:
        country_data = scenario_df[scenario_df['Country'] == country]
        for year in years:
            try:
                gdp_data = country_data[country_data['Variable'] == 'gdp']
                inflation_data = country_data[country_data['Variable'] == 'inflation']
                unemployment_data = country_data[country_data['Variable'] == 'unemployment']

                if gdp_data.empty or inflation_data.empty or unemployment_data.empty:
                    continue

                gdp = float(gdp_data[year].iloc[0])
                inflation = float(inflation_data[year].iloc[0])
                unemployment = float(unemployment_data[year].iloc[0])

                X_pred = pd.DataFrame({
                    'inflation': [inflation],
                    'unemployment': [unemployment],
                    'gdp': [gdp],
                    'is_solar': [tech_dummies['is_solar']],
                    'is_wind_onshore': [tech_dummies['is_wind_onshore']],
                    'is_wind_offshore': [tech_dummies['is_wind_offshore']]
                })

                wacc_pred = model.predict(X_pred)[0]
                forecasts.append({
                    'Country': country,
                    'ISO': country_data['ISO'].iloc[0],
                    'Year': year,
                    'Technology': tech_name,
                    'WACC': wacc_pred
                })

            except Exception:
                continue

    return pd.DataFrame(forecasts)

# === Forecast WACC ===
tech_combinations = {
    "Solar PV": {"is_solar": 1, "is_wind_onshore": 0, "is_wind_offshore": 0},
    "Wind Onshore": {"is_solar": 0, "is_wind_onshore": 1, "is_wind_offshore": 0},
    "Wind Offshore": {"is_solar": 0, "is_wind_onshore": 0, "is_wind_offshore": 1}
}

wacc_forecasts_all = []
for scenario in scenario_names:
    scenario_df = scenario_dfs[scenario]
    for tech_name, tech_dummies in tech_combinations.items():
        forecast_df = forecast_wacc(model, scenario_df, years, tech_name, tech_dummies)
        if not forecast_df.empty:
            forecast_df['Scenario'] = scenario
            wacc_forecasts_all.append(forecast_df)

# === Final Combined Forecasts ===
if wacc_forecasts_all:
    wacc_forecast_full = pd.concat(wacc_forecasts_all, ignore_index=True)
    # Export to CSV or just print head
wacc_forecast_full.to_csv("wacc_forecast_by_scenario.csv", index=False)
print("\n✅ Forecasts saved to: wacc_forecast_by_scenario.csv")
print(wacc_forecast_full.head())

#rename the scenarios here as well:
wacc_forecast_full['Scenario'] = wacc_forecast_full['Scenario'].replace(scenario_name_map)

# === Helper Function to Convert Country Names to ISO Codes ===
def get_iso_alpha3(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

# === File Paths ===
macro_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data1/macrofinal.csv"
wacc_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/merged_with_wacc_updated.csv"

# === Load Data ===
macro_df = pd.read_csv(macro_path)
wacc_df = pd.read_csv(wacc_path)

# === Clean Macrofinal ===
macro_df["Country"] = macro_df["Region"].str.replace(r"^NiGEM NGFS v1\.24\.2\|\s*", "", regex=True)

regions_to_exclude = [
    "Africa", "Asia", "Central America", "Europe", "Middle East", "North America",
    "South America", "World", "Rest of the World", "European Union", "Pacific Island States"
]
macro_df = macro_df[~macro_df["Country"].isin(regions_to_exclude)].copy()

# Normalize variable names
variable_rename_map = {
    "Unemployment rate ; %": "unemployment",
    "Inflation rate ; %": "inflation",
    "Gross Domestic Product (GDP)": "gdp",
    "Unemployment rate ; %(combined)": "unemployment",
    "Inflation rate ; %(combined)": "inflation",
    "Gross Domestic Product (GDP)(combined(no bus))": "gdp"
}
macro_df["Variable"] = macro_df["Variable"].map(variable_rename_map)
macro_df = macro_df[macro_df["Variable"].notna()].copy()

# ISO Conversion
macro_df["ISO"] = macro_df["Country"].apply(get_iso_alpha3)
manual_iso_map = {
    "Russia": "RUS",
    "South Korea": "KOR"
}
macro_df["ISO"] = macro_df.apply(lambda row: manual_iso_map.get(row["Country"], row["ISO"]), axis=1)
macro_df = macro_df[macro_df["ISO"].notna()].copy()

# === Merge with WACC data ===
merged_macro_wacc = pd.merge(
    macro_df,
    wacc_df,
    on="ISO",
    suffixes=("_macro", "_wacc")
)

# === Train Regression Model ===
features = ["inflation", "unemployment", "gdp", "is_solar", "is_wind_onshore", "is_wind_offshore"]
X = merged_macro_wacc[features]
y = merged_macro_wacc["wacc"]
X = X.dropna()
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# === Rebuild Scenarios ===
baseline_df = macro_df[macro_df["Scenario"] == "Baseline"].copy()
scenario_names = macro_df["Scenario"].dropna().unique()
scenario_names = [s for s in scenario_names if s != "Baseline"]
years = [str(y) for y in range(2022, 2051)]

scenario_dfs = {}
for scenario in scenario_names:
    scenario_df = macro_df[macro_df["Scenario"] == scenario].copy()
    merged = pd.merge(
        scenario_df,
        baseline_df,
        on=["Country", "Variable", "ISO"],
        suffixes=("_diff", "_base")
    )
    for year in years:
        merged[year] = merged[f"{year}_base"] + merged[f"{year}_diff"]
    final_cols = ["Scenario_diff", "Country", "Variable", "ISO"] + years
    scenario_dfs[scenario] = merged[final_cols].rename(columns={"Scenario_diff": "Scenario"})

# === Forecast Function ===
def forecast_wacc(model, scenario_df, years, tech_name, tech_dummies):
    forecasts = []
    countries = scenario_df['Country'].unique()

    for country in countries:
        country_data = scenario_df[scenario_df['Country'] == country]
        for year in years:
            try:
                gdp_data = country_data[country_data['Variable'] == 'gdp']
                inflation_data = country_data[country_data['Variable'] == 'inflation']
                unemployment_data = country_data[country_data['Variable'] == 'unemployment']

                if gdp_data.empty or inflation_data.empty or unemployment_data.empty:
                    continue

                gdp = float(gdp_data[year].iloc[0])
                inflation = float(inflation_data[year].iloc[0])
                unemployment = float(unemployment_data[year].iloc[0])

                X_pred = pd.DataFrame({
                    'inflation': [inflation],
                    'unemployment': [unemployment],
                    'gdp': [gdp],
                    'is_solar': [tech_dummies['is_solar']],
                    'is_wind_onshore': [tech_dummies['is_wind_onshore']],
                    'is_wind_offshore': [tech_dummies['is_wind_offshore']]
                })

                wacc_pred = model.predict(X_pred)[0]
                forecasts.append({
                    'Country': country,
                    'ISO': country_data['ISO'].iloc[0],
                    'Year': year,
                    'Technology': tech_name,
                    'WACC': wacc_pred
                })

            except Exception:
                continue

    return pd.DataFrame(forecasts)

# === Forecast WACC ===
tech_combinations = {
    "Solar PV": {"is_solar": 1, "is_wind_onshore": 0, "is_wind_offshore": 0},
    "Wind Onshore": {"is_solar": 0, "is_wind_onshore": 1, "is_wind_offshore": 0},
    "Wind Offshore": {"is_solar": 0, "is_wind_onshore": 0, "is_wind_offshore": 1}
}

wacc_forecasts_all = []
for scenario in scenario_names:
    scenario_df = scenario_dfs[scenario]
    for tech_name, tech_dummies in tech_combinations.items():
        forecast_df = forecast_wacc(model, scenario_df, years, tech_name, tech_dummies)
        if not forecast_df.empty:
            forecast_df['Scenario'] = scenario
            wacc_forecasts_all.append(forecast_df)

# === Final Combined Forecasts ===
if wacc_forecasts_all:
    wacc_forecast_full = pd.concat(wacc_forecasts_all, ignore_index=True)
    print("\nWACC Forecast by Scenario:")
    print("-" * 50)
    # Display summary statistics
    print("\nSummary Statistics:")
    print(wacc_forecast_full.groupby(['Scenario', 'Technology'])['WACC'].describe())

    # Display the first few rows of the forecast
    print("\nSample of Forecast Results:")
    print(wacc_forecast_full.head())

    # Save to CSV for full results
    output_file = 'wacc_forecast_results.csv'
    wacc_forecast_full.to_csv(output_file, index=False)
    print(f"\nFull results have been saved to '{output_file}'")
else:
    print("❌ No forecasts generated. Please check data integrity.")

# Count unique scenarios in the final forecast output DataFrame
if 'wacc_forecast_full' in globals():
    unique_scenarios_forecast = wacc_forecast_full['Scenario'].nunique()
    unique_scenarios_list_forecast = wacc_forecast_full['Scenario'].unique().tolist()

    print(f"\nNumber of unique scenarios in final forecast output: {unique_scenarios_forecast}")
    print(f"Scenarios in final forecast output: {unique_scenarios_list_forecast}")
else:
    print("Final forecast DataFrame 'wacc_forecast_full' not found.")
