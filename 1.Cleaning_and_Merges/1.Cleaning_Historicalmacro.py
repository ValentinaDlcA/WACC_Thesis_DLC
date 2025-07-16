#Merging RAW data from WorldBank database with WACC RAW: creates "merged_with_wacc.csv"

import pandas as pd
from pathlib import Path
import country_converter as coco

# 1. Loading data:
base_path = Path('data1_old')
files = {
    'inflation': base_path / '/Users/valentinadlc/PyCharmMiscProject/WACC_Thesis/Data/data1_old/Inflation/Inflation.csv',
    'gdp': base_path / '/Users/valentinadlc/PyCharmMiscProject/WACC_Thesis/Data/data1_old/GDP/gdp_capita.csv',
    'interest_rate': base_path / '/Users/valentinadlc/PyCharmMiscProject/WACC_Thesis/Data/data1_old/Interest_rate/Interest_rate.csv',
    'unemployment': base_path / '/Users/valentinadlc/PyCharmMiscProject/WACC_Thesis/Data/data1_old/Unemployment/Unemployment.csv',
    'population': base_path / '/Users/valentinadlc/PyCharmMiscProject/WACC_Thesis/Data/data1_old/Population/Population.csv'
}

def load_and_melt(filepath, variable_name):
    # Convert Path object to string and check if file exists
    filepath_str = str(filepath)
    if not Path(filepath_str).exists():
        raise FileNotFoundError(f"The file {filepath_str} does not exist. Please check the path.")
        
    # Explicitly specify delimiter as comma and skip the first 4 rows
    df = pd.read_csv(filepath_str, delimiter=',', skiprows=4)
    
    # Rename first column to 'country' and second to 'ISO'
    df.rename(columns={df.columns[0]: 'country', df.columns[1]: 'ISO'}, inplace=True)
    
    # Melt year columns into rows
    df_melted = df.melt(id_vars=['ISO'], var_name='year', value_name=variable_name)
    
    # Clean year column
    df_melted['year'] = pd.to_numeric(df_melted['year'], errors='coerce')
    df_melted = df_melted.dropna(subset=['year'])
    df_melted['year'] = df_melted['year'].astype(int)
    
    return df_melted

# Load and transform datasets
inflation_df = load_and_melt(files['inflation'], 'inflation')
gdp_df = load_and_melt(files['gdp'], 'gdp')
interest_df = load_and_melt(files['interest_rate'], 'interest_rate')
unemployment_df = load_and_melt(files['unemployment'], 'unemployment')  # ✅ fixed
population_df = load_and_melt(files['population'], 'population')



# Merge all dataframes on country and year
df = inflation_df \
    .merge(gdp_df, on=['ISO', 'year'], how='outer') \
    .merge(interest_df, on=['ISO', 'year'], how='outer') \
    .merge(unemployment_df, on=['ISO', 'year'], how='outer') \
    .merge(population_df, on=['ISO', 'year'], how='outer')

# Optional: Convert all values to numeric
for col in ['inflation', 'gdp', 'interest_rate', 'unemployment', 'population']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where either iso or year is missing
df = df.dropna(subset=['ISO', 'year'])

# Save to CSV
df.to_csv("merged_macro_data.csv", index=False)

# Preview
print("✅ Final merged dataset:")
print(df.head())

#Add WACC data set
# Load macroeconomic dataset
macro = pd.read_csv("/1.Cleaning_and_Merges/merged_macro_data.csv")
# macro['country'] = macro['country'].str.strip().str.lower()
macro['year'] = pd.to_numeric(macro['year'], errors='coerce')

# Load WACC dataset
wacc = pd.read_excel("/Users/valentinadlc/Documents/MASTER/MASTER THESIS/WACC_Thesis_DLC/Data/data_projection_df/WACC.xlsx", engine="openpyxl")

# ✅ Keep only the columns we care about (using the actual names from the Excel file)
wacc = wacc[['Country name', 'Technology', 'Financing year', 'WACC (nominal, after-tax)']]

# ✅ Rename columns to standardize
wacc.rename(columns={
    'Country name': 'country',
    'Technology': 'technology',
    'Financing year': 'year',
    'WACC (nominal, after-tax)': 'wacc'
}, inplace=True)

# Clean strings and convert year
wacc['country'] = wacc['country'].str.strip().str.lower()
wacc['technology'] = wacc['technology'].str.strip().str.lower()
wacc['year'] = pd.to_numeric(wacc['year'], errors='coerce')

#add country ISO as column
wacc['ISO'] = wacc['country'].apply(lambda x: coco.convert(x, to='ISO3'))
print(wacc.head())

# ✅ Merge WACC with macro data
merged = wacc.merge(macro, on=['ISO', 'year'], how='left')
merged['country'] = merged['ISO'].apply(lambda x: coco.convert(x, to='name_official'))

# Save result
merged.to_csv("merged_with_wacc.csv", index=False)

print("✅ Merged WACC (AT) with macro data:")
print(merged.head())

#How many countries : final count is 83 countries
print(merged['country'].nunique())
#to excel
# Export merged DataFrame to Excel
#merged.to_excel("merged_with_wacc.xlsx", index=False)
#print("✅ Merged dataset exported to Excel as 'merged_with_wacc.xlsx'")