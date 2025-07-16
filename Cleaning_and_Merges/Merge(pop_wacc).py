import pandas as pd

# File paths (adjust if needed)
pop_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data1/Population/Population.csv"
wacc_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data1/wacc_with_gdpppp2021.csv"

# Load the Population file, skipping the first 4 rows
pop_df_raw = pd.read_csv(pop_path, skiprows=4)

# Load the WACC dataframe
wacc_df = pd.read_csv(wacc_path)

# Step 1: Keep only "Country Code" and years from 2008 to 2023
years = [str(year) for year in range(2008, 2024)]
pop_df_filtered = pop_df_raw[['Country Code'] + years]

# Step 2: Reshape from wide to long format
pop_long = pop_df_filtered.melt(
    id_vars='Country Code',
    var_name='Year',
    value_name='Population'
)

# Step 3: Rename column for merging
pop_long.rename(columns={'Country Code': 'ISO'}, inplace=True)

# Step 4: Convert "Year" column to integers
pop_long['Year'] = pop_long['Year'].astype(int)

# Step 5: Merge WACC with Population using ISO and Year
merged_df = pd.merge(wacc_df, pop_long, on=['ISO', 'Year'], how='left')

# (Optional) Show or export the merged DataFrame
print(merged_df.head())

#Convert GDP_PPP from trillions to billions
# Create a new column for GDP_PPP in billions
merged_df['gdp_ppp'] = merged_df['GDP_PPP'] / 1_000_000_000

# Optional: view the result
print(merged_df[['ISO', 'Year', 'GDP_PPP', 'gdp_ppp']].head())

# Drop the GDP_PPP column
merged_df.drop(columns=['GDP_PPP'], inplace=True)

# Optional: view the result
print(merged_df.head())

# Save the final DataFrame to CSV
merged_df.to_csv("final_wacc_macro_historical.csv", index=False)