import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
file_path = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/wacc_projection_FE_nopop_with_groups.csv"
df = pd.read_csv(file_path)
df['wacc_projection'] *= 100  # Convert to percentage
df_filtered = df[df['Year'].isin([2025, 2050])]

# === Create output folder ===
output_dir = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/Plots_FE_BARS"
os.makedirs(output_dir, exist_ok=True)

# Build country maps
region_country_map = df.groupby('Region')['Country'].unique().apply(lambda x: ', '.join(sorted(set(x))))
income_country_map = df.groupby('IncomeLevel')['Country'].unique().apply(lambda x: ', '.join(sorted(set(x))))

# === REGION PLOTS ===
region_grouped = df_filtered.groupby(['Region', 'Technology', 'Scenario', 'Year'])['wacc_projection'].mean().reset_index()

for scenario in region_grouped['Scenario'].unique():
    for tech in region_grouped['Technology'].unique():
        subset = region_grouped[(region_grouped['Scenario'] == scenario) & (region_grouped['Technology'] == tech)]
        pivot = subset.pivot(index='Region', columns='Year', values='wacc_projection').sort_values(by=2025, ascending=False)

        # Truncate label text to improve readability
        pivot.index = [f"{idx} ({region_country_map[idx][:40]}...)" for idx in pivot.index]

        ax = pivot.plot(kind='barh', figsize=(12, 8), color={2025: 'gray', 2050: 'blue'})
        plt.title(f"WACC by Region – {tech} – {scenario}")
        plt.xlabel("WACC (%)")
        plt.ylabel("Region")
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=8)
        plt.subplots_adjust(left=0.4)  # more space for y-axis labels
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

# === INCOME LEVEL PLOTS ===
income_grouped = df_filtered.groupby(['IncomeLevel', 'Technology', 'Scenario', 'Year'])['wacc_projection'].mean().reset_index()

for scenario in income_grouped['Scenario'].unique():
    for tech in income_grouped['Technology'].unique():
        subset = income_grouped[(income_grouped['Scenario'] == scenario) & (income_grouped['Technology'] == tech)]
        pivot = subset.pivot(index='IncomeLevel', columns='Year', values='wacc_projection').sort_values(by=2025, ascending=False)

        pivot.index = [f"{idx} ({income_country_map[idx][:40]}...)" for idx in pivot.index]

        ax = pivot.plot(kind='barh', figsize=(12, 6), color={2025: 'gray', 2050: 'blue'})
        plt.title(f"WACC by Income Level – {tech} – {scenario}")
        plt.xlabel("WACC (%)")
        plt.ylabel("Income Group")
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=8)
        plt.subplots_adjust(left=0.4)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
