import pandas as pd
import matplotlib.pyplot as plt
import os

# Set output directory and ensure it exists
output_dir = "/Users/valentinadlc/PyCharmMiscProject/WACC_Thesis/Plots/Plots_country_OLS_FE"
os.makedirs(output_dir, exist_ok=True)

# Load data
ols_df = pd.read_csv("/Users/valentinadlc/Documents/MASTER/MASTER THESIS/WACC_Thesis_DLC/3.Projection/wacc_projection_OLS_with_groups.csv")
fe_df = pd.read_csv("/Users/valentinadlc/Documents/MASTER/MASTER THESIS/WACC_Thesis_DLC/3.Projection/wacc_projection_FE_nopop_with_groups.csv")

# Label each dataset
fe_df["Model"] = "FE"
ols_df["Model"] = "OLS"

# Combine datasets
combined_df = pd.concat([fe_df, ols_df])

# Filter for specific technologies
filtered_data = combined_df[
    combined_df["Technology"].isin(["Wind_Onshore", "Solar_PV", "Wind_Offshore"])
]

# Choose a specific year to plot
target_year = 2050  # You can change this to 2025, 2030, etc.

# Filter to selected year
filtered_data = filtered_data[filtered_data["Year"] == target_year]

# Loop and create bar plots by country (no averaging)
for model in ["FE", "OLS"]:
    model_df = filtered_data[filtered_data["Model"] == model]

    for scenario in model_df["Scenario"].unique():
        scenario_df = model_df[model_df["Scenario"] == scenario]

        for tech in scenario_df["Technology"].unique():
            tech_df = scenario_df[scenario_df["Technology"] == tech]

            # Sort values for cleaner plot
            tech_df = tech_df.sort_values(by="wacc_projection", ascending=False)

            # Plot bar chart
            plt.figure(figsize=(12, 6))
            plt.bar(tech_df["Country"], tech_df["wacc_projection"])
            plt.title(f"Projected WACC by Country in {target_year}\nTechnology: {tech} | Scenario: {scenario} | Model: {model}")
            plt.xlabel("Country")
            plt.ylabel("WACC")
            plt.xticks(rotation=90)
            plt.grid(axis='y')

            # Set detailed y-axis ticks
            plt.yticks([i / 100 for i in range(-10, 15)])  # From -0.10 to 0.14

            plt.tight_layout()

            # Save plot
            filename = f"wacc_bar_{tech}_{scenario}_{model}_{target_year}.png".replace("/", "-")
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")

            plt.show()
