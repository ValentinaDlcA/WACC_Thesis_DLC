import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
ols_df = pd.read_csv("/Users/valentinadlc/PyCharmMiscProject/WACC_Thesis/Data/wacc_projection_OLS_with_groups.csv")
fe_df = pd.read_csv("/Users/valentinadlc/PyCharmMiscProject/WACC_Thesis/Data/wacc_projection_FE_nopop_with_groups.csv")

# Add model labels
ols_df["Model"] = "OLS"
fe_df["Model"] = "FE"

# Combine datasets
combined_df = pd.concat([ols_df, fe_df], ignore_index=True)

# Filter for Germany
germany_df = combined_df[combined_df["Country"] == "Germany"]

# Define output directory
output_dir = "/Users/valentinadlc/PyCharmMiscProject/WACC_Thesis/Plots/Plots_Germany2"
os.makedirs(output_dir, exist_ok=True)

# Set plot style
sns.set(style="whitegrid")

# Function to create and save bar plots per technology
def plot_germany_wacc_by_scenario_and_model(df):
    for tech in df["Technology"].unique():
        tech_df = df[df["Technology"] == tech]

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=tech_df,
            x="Scenario",
            y="wacc_projection",
            hue="Model",
            dodge=True,
            palette="muted"
        )

        plt.title(f"Projected WACC – Germany – {tech}")
        plt.xlabel("Scenario")
        plt.ylabel("Projected WACC")
        plt.xticks(rotation=45)
        plt.legend(title="Model")
        plt.tight_layout()

        # Save plot
        filename = f"WACC_Germany_{tech.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()

# Call the function
plot_germany_wacc_by_scenario_and_model(germany_df)
