import pandas as pd

# === Step 1: Load both datasets ===
macro_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data1/cleaned_macro_data_pivoted.csv"
popgdp_path = "/1.Cleaning_and_Merges/popgdp_cleaned.csv"

df_macro = pd.read_csv(macro_path)
df_popgdp = pd.read_csv(popgdp_path)

# === Step 2: Rename 'partner_iso' to 'ISO' in macro data (just in case it still exists) ===
if 'partner_iso' in df_macro.columns:
    df_macro = df_macro.rename(columns={"partner_iso": "ISO"})

# === Step 3: Pivot selected variables from popgdp dataset ===
# Filter for relevant variables
relevant_vars = ['Population', 'GDP_damage', 'GDP_no_damage']
df_popgdp_filtered = df_popgdp[df_popgdp['Variable'].isin(relevant_vars)]

# Pivot so each variable becomes a column
df_popgdp_pivoted = df_popgdp_filtered.pivot_table(
    index=["Model", "Scenario", "ISO", "Year"],
    columns="Variable",
    values="Value"
).reset_index()

# === Step 4: Merge on ISO, Scenario, Year (and Model if needed) ===
# If Model is not used for merge, drop it from popgdp
df_merged = pd.merge(
    df_macro,
    df_popgdp_pivoted,
    on=["ISO", "Scenario", "Year"],
    how="left"
)

# === Step 5: Save merged dataset ===
output_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data1/merged_macro_popgdp.csv"
df_merged.to_csv(output_path, index=False)

# === Step 6: Preview the result ===
print(df_merged.head())
print("Merged Columns:", df_merged.columns.tolist())
