import pandas as pd
import re

# Load your local CSV
file_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/data1/macrofinal.csv"
df = pd.read_csv(file_path)

# Drop rows where Scenario is missing
df = df[df["Scenario"].notna()].copy()

# Clean variable names (remove things like "(combined)")
def clean_variable_name(var):
    if not isinstance(var, str):
        return ""
    return re.sub(r"\s*\(.*?\)", "", var).strip()

df["Variable"] = df["Variable"].apply(clean_variable_name)

# Clean and rename columns
df.rename(columns={"Region": "Country"}, inplace=True)
df["Country"] = df["Country"].str.replace(r"^NiGEM NGFS v1\.24\.2\|\s*", "", regex=True)
df.drop(columns=["Model"], inplace=True, errors="ignore")

# Rename variable names for simplicity
variable_renaming = {
    "Gross Domestic Product": "GDP",
    "Unemployment rate ; %": "unemployment",
    "Inflation rate ; %": "inflation"
}
df["Variable"] = df["Variable"].replace(variable_renaming)

# Separate baseline and scenarios
baseline_df = df[df["Scenario"] == "Baseline"].copy()
non_baseline_scenarios = df["Scenario"].unique()
non_baseline_scenarios = [s for s in non_baseline_scenarios if s != "Baseline"]
years = [str(y) for y in range(2022, 2051)]

# Compute scenario values
scenario_dfs = {}

for scenario in non_baseline_scenarios:
    scenario_df = df[df["Scenario"] == scenario].copy()
    merged = pd.merge(
        scenario_df,
        baseline_df,
        on=["Country", "Variable"],
        suffixes=("_diff", "_base")
    )
    for year in years:
        merged[year] = merged[f"{year}_base"] + merged[f"{year}_diff"]
    final_cols = ["Scenario_diff", "Country", "Variable"] + years
    result = merged[final_cols].rename(columns={"Scenario_diff": "Scenario"})
    scenario_dfs[scenario] = result

# OPTIONAL: Verify scenario calculation
check = []

for i in range(5):  # Check first 5 entries
    row = baseline_df.iloc[i]
    match = scenario_dfs["Below 2?C"][
        (scenario_dfs["Below 2?C"]["Country"] == row["Country"]) &
        (scenario_dfs["Below 2?C"]["Variable"] == row["Variable"])
    ]
    if not match.empty:
        year = "2023"
        baseline_val = row[year]
        scenario_val = match[year].values[0]
        computed_diff = scenario_val - baseline_val
        check.append({
            "Country": row["Country"],
            "Variable": row["Variable"],
            "Baseline_2023": baseline_val,
            "Scenario_2023": scenario_val,
            "Computed_Diff": computed_diff
        })

# Print verification check
check_df = pd.DataFrame(check)
print(check_df)

# Loop through and print the first few rows of each scenario DataFrame
for scenario_name, df in scenario_dfs.items():
    print(f"\n===== Scenario: {scenario_name} =====")
    print(df.head())  # You can use df.to_string(index=False) for full formatting