README – WACC_Thesis_DLC Project Overview

This project is organized into five main directories, each serving a specific purpose in the workflow for regression and projection analysis related to WACC (Weighted Average Cost of Capital) and macroeconomic/scenario-based modeling.

1. Cleaning_and_Merges

This folder contains all data preparation scripts. Files are ordered numerically and include descriptive titles. They progressively clean, merge, and format datasets needed for regression and projections.

Key Files:
1. Cleaning_Historicalmacro.py – Merges World Bank macro data with WACC raw data.  
   Output: merged_with_wacc.csv

2. GDP_PPP_cleaning.py – Cleans latest GDP PPP raw data.

3. NGFS_cleaning.py – Cleans NIGEM model variables from the NGFS macro dataset.

4. popgdp_cleaned.py – Cleans population and GDP from the RemindMagpie model.

5. New_GDP_2021_to_billions_WACC.py – Converts GDP data to 2021 constant USD (billions).  
   Output: wacc_with_gdpppp2021.csv

6. Final_for_reg_Merge_(pop_wacc).py – Creates final dataset for regression (includes population and WACC).

7. Final_GDPPPP_Clean_plus_baselinecalc.py – Calculates GDP deviations from NGFS baseline.  
   Output: macro_gdp_merged.csv

8. Final_Merge_pop_ngfs_macro.py – Produces ngfs_final_merge.csv for projection models.

9. SSPS.py – Cleans SSPS governance, rule-of-law, and urbanization raw datasets.

10. Final_SSPS_WACC.py – Merges all SSPS for regression models using SSPS variables and WACC.

Note: One extra Python file is included that was used for testing alternative variables. It is not central to the analysis.

All key DataFrames created in this phase are stored for reuse in regression and projections.

2. Regression

This folder contains all regression model scripts using cleaned macroeconomic and SSPS datasets.

Key Files:
1. FINAL_Reg_OLS_FE_RE_nostand.py – Contains all regression models (OLS, FE, RE) using final_wacc_macro_historical.csv, without standardizing variables. Includes full statistical testing.

2. FINAL_SSPS_REG_nostand.py – Regression models using SSPS variables (governance, urbanization, rule of law), also without standardization.

Other scripts in this folder include older versions and models with standardized variables.

3. Projection

This folder contains scripts that project WACC based on the regression model results.

Key Files:
1. Final_Projection_FE_nostandnopop.py – Projects NGFS variables using the FE model (excluding population), without standardized variables.

2. Final_Projection_OLS_nostandnopop.py – Projects NGFS variables using the OLS model, without standardized variables.

3. Final_Regions.py – Groups final projection outputs by region and World Bank income level, producing:
   - wacc_projection_FE_nopop_with_groups.csv
   - wacc_projection_OLS_with_groups.csv

4. Plots

Scripts for visualizing the regression and projection results.

Key Files:
1. Plots_region_income.py – Plots OLS and FE model outputs by region and income level.

2. Plots_country.py – Bar plots showing WACC results by country from both models.

3. Plot_germany.py – Focused plot results for Germany.

5. Data

This folder includes raw inputs and intermediate data files.

Subfolders:
- data1_old/ – Contains historical macro data from World Bank and NGFS, as well as raw SSPS data. Also includes some legacy DataFrames from a previous Python environment (these can be ignored).

- data_projection_df/ – Contains the raw Excel file for WACC and some duplicate files for World Bank variables.

Other:
- WACC_Thesis_old/ – Legacy folder copied from a previous Python environment. Not used in current workflow and can be ignored.
