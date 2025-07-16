import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

# ------------------------------------------------------------
# 1. Paths and Setup
# ------------------------------------------------------------
models = {
    "OLS": "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/wacc_projection_OLS_with_groups.csv",
    "FE_nopop": "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/wacc_projection_FE_nopop_with_groups.csv"
}

output_folder = "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/demo/PyCharmLearningProject/WACC Thesis/plots_combined"
os.makedirs(output_folder, exist_ok=True)

# ------------------------------------------------------------
# 2. Process each model
# ------------------------------------------------------------
for model_name, path in models.items():
    print(f"\n=== Processing model: {model_name} ===")
    df = pd.read_csv(path)

    df.columns = [c.strip() for c in df.columns]
    df["Year"] = df["Year"].astype(int)

    technologies = df["Technology"].unique()
    scenarios = df["Scenario"].unique()

    for tech in technologies:
        df_tech = df[df["Technology"] == tech].copy()

        # --- Plot by Scenario → grouped by Region
        for scen in scenarios:
            df_scen = df_tech[df_tech["Scenario"] == scen].copy()

            regional_wacc = (
                df_scen.groupby(["Year", "Region"])["wacc_projection"]
                .mean()
                .reset_index()
                .pivot(index="Year", columns="Region", values="wacc_projection")
            )

            fig, ax = plt.subplots(figsize=(12, 7))
            regional_wacc.plot(ax=ax, linewidth=2.5, marker='o')

            ax.set_ylim(0.05, None)
            ax.yaxis.set_major_locator(mtick.MultipleLocator(0.01))
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

            ax.set_xlabel("Year")
            ax.set_ylabel("Projected WACC")
            ax.set_title(f"[{model_name}] {tech} – Scenario: {scen}")
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.legend(title="Region", bbox_to_anchor=(1.02, 1), loc="upper left")

            plt.tight_layout()
            filename = f"{output_folder}/{model_name}_{tech}_Scenario_{scen}_by_Region.png"
            plt.savefig(filename)
            print(f"✅ Saved: {filename}")
            plt.close()

        # --- Plot by Region → grouped by Scenario
        for region in df_tech["Region"].dropna().unique():
            df_region = df_tech[df_tech["Region"] == region].copy()

            scenario_wacc = (
                df_region.groupby(["Year", "Scenario"])["wacc_projection"]
                .mean()
                .reset_index()
                .pivot(index="Year", columns="Scenario", values="wacc_projection")
            )

            fig, ax = plt.subplots(figsize=(12, 7))
            scenario_wacc.plot(ax=ax, linewidth=2.5, marker='o')

            ax.set_ylim(0.05, None)
            ax.yaxis.set_major_locator(mtick.MultipleLocator(0.01))
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

            ax.set_xlabel("Year")
            ax.set_ylabel("Projected WACC")
            ax.set_title(f"[{model_name}] {tech} – Region: {region}")
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.legend(title="Scenario", bbox_to_anchor=(1.02, 1), loc="upper left")

            plt.tight_layout()
            filename = f"{output_folder}/{model_name}_{tech}_Region_{region}_by_Scenario.png"
            plt.savefig(filename)
            print(f"✅ Saved: {filename}")
            plt.close()

        # --- Plot by Scenario → grouped by IncomeLevel
        for scen in scenarios:
            df_scen = df_tech[df_tech["Scenario"] == scen].copy()

            income_wacc = (
                df_scen.groupby(["Year", "IncomeLevel"])["wacc_projection"]
                .mean()
                .reset_index()
                .pivot(index="Year", columns="IncomeLevel", values="wacc_projection")
            )

            fig, ax = plt.subplots(figsize=(12, 7))
            income_wacc.plot(ax=ax, linewidth=2.5, marker='o')

            ax.set_ylim(0.05, None)
            ax.yaxis.set_major_locator(mtick.MultipleLocator(0.01))
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

            ax.set_xlabel("Year")
            ax.set_ylabel("Projected WACC")
            ax.set_title(f"[{model_name}] {tech} – Scenario: {scen} by Income Level")
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.legend(title="IncomeLevel", bbox_to_anchor=(1.02, 1), loc="upper left")

            plt.tight_layout()
            filename = f"{output_folder}/{model_name}_{tech}_Scenario_{scen}_by_IncomeLevel.png"
            plt.savefig(filename)
            print(f"✅ Saved: {filename}")
            plt.close()

    # ------------------------------------------------------------
    # 3. SUMMARY TABLES
    # ------------------------------------------------------------
    key_years = [2025, 2030, 2050]

    summary_region = (
        df.loc[df["Year"].isin(key_years)]
        .groupby(["Technology", "Scenario", "Region", "Year"])["wacc_projection"]
        .mean()
        .round(4)
        .unstack("Year")
        .sort_index()
    )

    summary_income = (
        df.loc[df["Year"].isin(key_years)]
        .groupby(["Technology", "Scenario", "IncomeLevel", "Year"])["wacc_projection"]
        .mean()
        .round(4)
        .unstack("Year")
        .sort_index()
    )

    summary_region_path = f"{output_folder}/{model_name}_summary_table_by_region.csv"
    summary_income_path = f"{output_folder}/{model_name}_summary_table_by_income.csv"

    summary_region.to_csv(summary_region_path)
    summary_income.to_csv(summary_income_path)

    print(f"\n✅ Summary (Region) saved to: {summary_region_path}")
    print(f"✅ Summary (IncomeLevel) saved to: {summary_income_path}")
