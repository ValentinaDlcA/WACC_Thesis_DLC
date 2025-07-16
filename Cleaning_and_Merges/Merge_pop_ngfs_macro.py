import pandas as pd

# Load the files
macro_gdp_merged = pd.read_csv("/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/macro_gdp_merged.csv")
popgdp_cleaned = pd.read_csv("/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data1/popgdp_cleaned.csv")

# Step 1: Rename scenarios
popgdp_cleaned['Scenario'] = popgdp_cleaned['Scenario'].replace({
    'Below 2C': 'Below2',
    'Nationally Determined Contributions (NDCs)': 'NDC',
    'Net Zero 2050': 'Netzero'
})

# Step 2: Filter only Population values from Remind-Magpie model
pop_filtered = popgdp_cleaned[
    (popgdp_cleaned['Model'] == 'Remind-Magpie') &
    (popgdp_cleaned['Variable'].str.contains('Population', case=False, na=False))
]

# Step 3: Pivot to extract Population as its own column
pop_pivoted = pop_filtered.pivot_table(
    index=['Scenario', 'ISO', 'Year'],
    columns='Variable',
    values='Value'
).reset_index()

# Step 4: Keep only the Population column and rename it to lowercase
pop_pivoted = pop_pivoted[['Scenario', 'ISO', 'Year', 'Population']]
pop_pivoted.rename(columns={'Population': 'population'}, inplace=True)

# Step 5: Merge with macro_gdp_merged
merged_df_with_population = pd.merge(
    macro_gdp_merged,
    pop_pivoted,
    on=['Scenario', 'ISO', 'Year'],
    how='left'
)

# Optional: Save or preview
# merged_df_with_population.to_csv("macro_gdp_with_population.csv", index=False)
print(merged_df_with_population.head())
#save
merged_df_with_population.to_csv("ngfs_final_merge.csv", index=False)