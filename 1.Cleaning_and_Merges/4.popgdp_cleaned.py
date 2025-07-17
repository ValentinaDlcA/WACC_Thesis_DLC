#cleans population and gdp from another NGFS model
import pandas as pd

# Load the dataset
file_path = "/Users/valentinadlc/Documents/MASTER/MASTER THESIS/WACC_Thesis_DLC/Data/data1_old/popgdp.csv"
df = pd.read_csv(file_path)

# Step 1: Clean "Model" column values
df["Model"] = df["Model"].replace({
    "Downscaling[REMIND-MAgPIE 3.3-4.8 IntegratedPhysicalDamages (median)]": "Remind-Magpie_damages",
    "Downscaling[REMIND-MAgPIE 3.3-4.8]": "Remind-Magpie"
})

# Step 2: Standardize "Scenario" values
df["Scenario"] = df["Scenario"].replace({"Below 2?C": "Below 2C"})

# Step 3: Rename "Region" to "ISO"
df = df.rename(columns={"Region": "ISO"})

# Step 4: Rename "Variable" values
df["Variable"] = df["Variable"].replace({
    "GDP|PPP|including medium chronic physical risk damage estimate": "GDP_damage",
    "GDP|PPP|Counterfactual without damage": "GDP_no_damage"
})

# Step 5: Drop columns "Unit" and "2020"
df = df.drop(columns=["Unit", "2020"], errors='ignore')

# Step 6: Melt year columns into rows
# Identify year columns (all remaining numeric columns)
year_columns = df.columns[df.columns.str.match(r'^\d{4}$')]
id_vars = [col for col in df.columns if col not in year_columns]

# Melt dataframe
df_melted = df.melt(id_vars=id_vars, var_name="Year", value_name="Value")

# Save cleaned dataset (optional)
output_path = "/1.Cleaning_and_Merges/popgdp_cleaned.csv"
df_melted.to_csv(output_path, index=False)

# Show preview
print(df_melted.head())
