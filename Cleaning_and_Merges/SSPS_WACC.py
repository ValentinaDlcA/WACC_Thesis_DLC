#MERGE SSPS AND WACC
import pandas as pd

# === Load CSV files ===
urban_df = pd.read_csv("/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/urbanization_pre2024.csv")
rule_df = pd.read_csv("/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/rule_law_pre2024.csv")
gov_df = pd.read_csv("/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/governance_1996_2024.csv")

# === Filter by scenario and variable ===
urban_clean = urban_df[
    (urban_df["Scenario"] == "SSP2") &
    (urban_df["Variable"] == "Population|Urban [Share]")
].copy()

rule_clean = rule_df[
    (rule_df["Scenario"] == "SSP2") &
    (rule_df["Variable"] == "Rule-of-Law Index")
].copy()

gov_clean = gov_df[
    (gov_df["Scenario"] == "Observed") &
    (gov_df["Variable"] == "Governance Index")
].copy()

# === Rename variable names for simplicity ===
urban_clean["Variable"] = "Urban_Share"
rule_clean["Variable"] = "Rule_of_Law"
gov_clean["Variable"] = "Governance_Index"

# === Pivot each dataset to wide format ===
urban_wide = urban_clean.pivot_table(
    index=["Country", "ISO", "Year"],
    columns="Variable", values="Value"
).reset_index()

rule_wide = rule_clean.pivot_table(
    index=["Country", "ISO", "Year"],
    columns="Variable", values="Value"
).reset_index()

gov_wide = gov_clean.pivot_table(
    index=["Country", "ISO", "Year"],
    columns="Variable", values="Value"
).reset_index()

# === Merge datasets on Country, ISO, and Year ===
merged_df = pd.merge(urban_wide, rule_wide, on=["Country", "ISO", "Year"], how="inner")
merged_df = pd.merge(merged_df, gov_wide, on=["Country", "ISO", "Year"], how="inner")

# === Save cleaned and merged dataset ===
output_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/cleaned_ssps_data.csv"
merged_df.to_csv(output_path, index=False)

print(f"✅ Cleaned dataset saved to:\n{output_path}")

#MERGE WITH WACC
import pandas as pd

# === Load SSPS cleaned dataset ===
ssps_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/cleaned_ssps_data.csv"
ssps_df = pd.read_csv(ssps_path)

# === Load WACC dataset ===
wacc_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/wacc_cleaned.csv"
wacc_df = pd.read_csv(wacc_path)

# === Step 1: Aggregate WACC to average per country-year ===
wacc_avg = wacc_df.groupby(['Country', 'ISO', 'Year'], as_index=False)['wacc'].mean()

# === Step 2: Merge with SSPS dataset === retain all technology-level rows
merged_final = pd.merge(wacc_df, ssps_df, on=["Country", "ISO", "Year"], how="inner")

# Save final dataset
merged_final.to_csv("final_merged_dataset.csv", index=False)
print("✅ Final merged dataset saved to 'final_merged_dataset.csv'")