import pandas as pd

# --- File paths ---
ols_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/wacc_projection_by_scenario.csv"
fe_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/wacc_projection_FE_nopop.csv"

# --- Load files ---
ols_df = pd.read_csv(ols_path)
fe_df = pd.read_csv(fe_path)

# --- ISO to metadata dictionaries ---
iso_to_country = {  # abbreviated for brevity
    "ARG": "Argentina", "AUS": "Australia", "AUT": "Austria", "BEL": "Belgium", "BGR": "Bulgaria",
    "BRA": "Brazil", "CAN": "Canada", "CHE": "Switzerland", "CHL": "Chile", "CHN": "China",
    "CZE": "Czech Republic", "DEU": "Germany", "DNK": "Denmark", "EGY": "Egypt", "ESP": "Spain",
    "EST": "Estonia", "FIN": "Finland", "FRA": "France", "GBR": "United Kingdom", "GRC": "Greece",
    "HKG": "Hong Kong", "HRV": "Croatia", "HUN": "Hungary", "IDN": "Indonesia", "IND": "India",
    "IRL": "Ireland", "ITA": "Italy", "JPN": "Japan", "KOR": "South Korea", "LTU": "Lithuania",
    "LVA": "Latvia", "MEX": "Mexico", "MYS": "Malaysia", "NLD": "Netherlands", "NOR": "Norway",
    "NZL": "New Zealand", "POL": "Poland", "PRT": "Portugal", "ROU": "Romania", "RUS": "Russia",
    "SGP": "Singapore", "SVK": "Slovakia", "SVN": "Slovenia", "SWE": "Sweden", "TWN": "Taiwan",
    "USA": "United States", "VNM": "Vietnam", "ZAF": "South Africa"
}

iso_to_region = {  # abbreviated for brevity
    "AUS": "East Asia and Pacific", "CHN": "East Asia and Pacific", "HKG": "East Asia and Pacific",
    "IDN": "East Asia and Pacific", "JPN": "East Asia and Pacific", "KOR": "East Asia and Pacific",
    "MYS": "East Asia and Pacific", "NZL": "East Asia and Pacific", "SGP": "East Asia and Pacific",
    "TWN": "East Asia and Pacific", "VNM": "East Asia and Pacific",
    "AUT": "Europe and Central Asia", "BEL": "Europe and Central Asia", "BGR": "Europe and Central Asia",
    "CHE": "Europe and Central Asia", "CZE": "Europe and Central Asia", "DEU": "Europe and Central Asia",
    "DNK": "Europe and Central Asia", "ESP": "Europe and Central Asia", "EST": "Europe and Central Asia",
    "FIN": "Europe and Central Asia", "FRA": "Europe and Central Asia", "GBR": "Europe and Central Asia",
    "GRC": "Europe and Central Asia", "HRV": "Europe and Central Asia", "HUN": "Europe and Central Asia",
    "IRL": "Europe and Central Asia", "ITA": "Europe and Central Asia", "LTU": "Europe and Central Asia",
    "LVA": "Europe and Central Asia", "NLD": "Europe and Central Asia", "NOR": "Europe and Central Asia",
    "POL": "Europe and Central Asia", "PRT": "Europe and Central Asia", "ROU": "Europe and Central Asia",
    "RUS": "Europe and Central Asia", "SVK": "Europe and Central Asia", "SVN": "Europe and Central Asia",
    "SWE": "Europe and Central Asia",
    "ARG": "Latin America and Caribbean", "BRA": "Latin America and Caribbean", "CHL": "Latin America and Caribbean",
    "MEX": "Latin America and Caribbean",
    "EGY": "Middle East and North Africa",
    "CAN": "North America", "USA": "North America",
    "IND": "South Asia",
    "ZAF": "Sub-Saharan Africa"
}

iso_to_income = {  # abbreviated for brevity
    "AUS": "High income", "AUT": "High income", "BEL": "High income", "CAN": "High income",
    "CHE": "High income", "CZE": "High income", "DEU": "High income", "DNK": "High income",
    "ESP": "High income", "EST": "High income", "FIN": "High income", "FRA": "High income",
    "GBR": "High income", "GRC": "High income", "HRV": "High income", "HUN": "High income",
    "IRL": "High income", "ITA": "High income", "JPN": "High income", "KOR": "High income",
    "LTU": "High income", "LVA": "High income", "NLD": "High income", "NOR": "High income",
    "NZL": "High income", "POL": "High income", "PRT": "High income", "SVK": "High income",
    "SVN": "High income", "SWE": "High income", "SGP": "High income", "TWN": "High income",
    "USA": "High income", "CHL": "High income",
    "ARG": "Upper middle income", "BRA": "Upper middle income", "CHN": "Upper middle income",
    "MEX": "Upper middle income", "RUS": "Upper middle income", "ZAF": "Upper middle income",
    "BGR": "Upper middle income", "ROU": "Upper middle income",
    "EGY": "Lower middle income", "IND": "Lower middle income", "IDN": "Lower middle income",
    "VNM": "Lower middle income",
    "HKG": "High income", "MYS": "Upper middle income"
}

# --- Mapping function ---
def enrich_with_metadata(df):
    df["Country"] = df["ISO"].map(iso_to_country)
    df["Region"] = df["ISO"].map(iso_to_region)
    df["IncomeLevel"] = df["ISO"].map(iso_to_income)

    # Reorder columns
    cols = df.columns.tolist()
    cols.remove("Country")
    cols.insert(cols.index("ISO") + 1, "Country")
    return df[cols]

# --- Apply enrichment ---
ols_df = enrich_with_metadata(ols_df)
fe_df = enrich_with_metadata(fe_df)

# --- Save both ---
ols_out = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/wacc_projection_OLS_with_groups.csv"
fe_out = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/wacc_projection_FE_nopop_with_groups.csv"

ols_df.to_csv(ols_out, index=False)
fe_df.to_csv(fe_out, index=False)

print("✅ Saved enriched OLS projection to:", ols_out)
print("✅ Saved enriched FE (no pop) projection to:", fe_out)
