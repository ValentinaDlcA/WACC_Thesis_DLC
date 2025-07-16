import pandas as pd
import matplotlib.pyplot as plt
import os

# Set output directory for plots
output_dir = "/Users/valentinadlc/PyCharmMiscProject/WACC_Thesis/Plots/Projection FE and OLS Plots"
os.makedirs(output_dir, exist_ok=True)

# Load both datasets
fe_df = pd.read_csv("/Users/valentinadlc/PyCharmMiscProject/WACC_Thesis/Data/wacc_projection_FE_nopop_with_groups.csv")
ols_df = pd.read_csv("/Users/valentinadlc/PyCharmMiscProject/WACC_Thesis/Data/wacc_projection_OLS_with_groups.csv")

# Label models
fe_df["Model"] = "FE"
ols_df["Model"] = "OLS"

# Combine datasets
combined_df = pd.concat([fe_df, ols_df])

# --- Compute average WACC per Region, Year, Scenario, Technology, Model ---
avg_by_region = (
    combined_df.groupby(["Region", "Year", "Scenario", "Technology", "Model"], as_index=False)
    .agg({"wacc_projection": "mean"})
)

# Filter for selected technologies (now including Wind_Offshore)
filtered_avg = avg_by_region[
    avg_by_region["Technology"].isin(["Wind_Onshore", "Solar_PV", "Wind_Offshore"])
]

# --- Generate plots by Region ---
technologies = filtered_avg["Technology"].unique()
scenarios = filtered_avg["Scenario"].unique()

for tech in technologies:
    for scen in scenarios:
        subset = filtered_avg[
            (filtered_avg["Technology"] == tech) &
            (filtered_avg["Scenario"] == scen)
        ]

        plt.figure(figsize=(10, 6))
        for model in ["FE", "OLS"]:
            model_data = subset[subset["Model"] == model]
            for region in model_data["Region"].unique():
                region_data = model_data[model_data["Region"] == region].sort_values("Year")
                label = f"{region} ({model})"
                linestyle = '-' if model == "FE" else '--'
                plt.plot(region_data["Year"], region_data["wacc_projection"],
                         label=label, linestyle=linestyle)

        plt.title(f"Projected WACC Over Time – {tech} – {scen}")
        plt.xlabel("Year")
        plt.ylabel("Average WACC")
        plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        plt.tight_layout()
        plt.grid(True)
        filename = f"wacc_region_{tech}_{scen}.png".replace("/", "-")
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.show()

# --- Compute average WACC by Income Level ---
avg_by_income = (
    combined_df.groupby(["IncomeLevel", "Year", "Scenario", "Technology", "Model"], as_index=False)
    .agg({"wacc_projection": "mean"})
)

# Filter for selected technologies (now including Wind_Offshore)
filtered_avg = avg_by_income[
    avg_by_income["Technology"].isin(["Wind_Onshore", "Solar_PV", "Wind_Offshore"])
]

# --- Generate plots by Income Level ---
technologies = filtered_avg["Technology"].unique()
scenarios = filtered_avg["Scenario"].unique()
models = ["FE", "OLS"]

for model in models:
    model_data = filtered_avg[filtered_avg["Model"] == model]

    for tech in technologies:
        for scen in scenarios:
            subset = model_data[
                (model_data["Technology"] == tech) &
                (model_data["Scenario"] == scen)
            ]

            plt.figure(figsize=(10, 6))
            for income in subset["IncomeLevel"].unique():
                income_data = subset[subset["IncomeLevel"] == income].sort_values("Year")
                plt.plot(income_data["Year"], income_data["wacc_projection"], label=income)

            plt.title(f"WACC Projection Over Time – {tech} – {scen} – Model: {model}")
            plt.xlabel("Year")
            plt.ylabel("Average WACC")
            plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
            plt.tight_layout()
            plt.grid(True)
            filename = f"wacc_income_{tech}_{scen}_{model}.png".replace("/", "-")
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.show()
