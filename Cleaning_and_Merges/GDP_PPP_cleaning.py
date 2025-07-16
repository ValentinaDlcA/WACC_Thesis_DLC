import pandas as pd

#CLEANING GDP-PPP LATEST RAW DATA
# Load the data
gdp_ppp_df = pd.read_csv('/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data/gdp_ppp.csv')

# Delete the first and 5th column
gdp_ppp_df = gdp_ppp_df.drop(columns=["Model", "Unit"])


# Print columns before renaming (for debugging)
print("Columns before renaming:", gdp_ppp_df.columns)

# Rename the region column to ISO
gdp_ppp_df = gdp_ppp_df.rename(columns={'Region': 'ISO'})  # Adjust to your actual column name!

# Print columns after renaming (for debugging)
print("Columns after renaming:", gdp_ppp_df.columns)

# Map scenario names
scenario_name_map = {
    "Below 2?C": "Below2",
    "Nationally Determined Contributions (NDCs)": "NDC",
    "Net Zero 2050": "Netzero"
}
gdp_ppp_df['Scenario'] = gdp_ppp_df['Scenario'].replace(scenario_name_map)

# List of regions to exclude
regions_to_exclude = [
    "Africa", "Asia", "Central America", "Europe", "Middle East", "North America",
    "South America", "World", "Rest of the World", "European Union", "Pacific Island States"
]
gdp_ppp_df = gdp_ppp_df[~gdp_ppp_df['ISO'].isin(regions_to_exclude)]

# Change variable name
gdp_ppp_df['Variable'] = gdp_ppp_df['Variable'].replace(
    "GDP|PPP|including medium chronic physical risk damage estimate",
    "gdp_ppp"
)

# Print the result to check
print(gdp_ppp_df.head())

#To csv
gdp_ppp_df.to_csv('/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data/gdp_ppp_cleaned.csv', index=False)