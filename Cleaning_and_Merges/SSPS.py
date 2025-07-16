"""
End-to-end cleaner for Governance, Urbanization and Rule-of-Law workbooks
Creates tidy long-format DataFrames with columns:
Scenario | Country | Variable | ISO | Year | Value
Splits them into requested year slices and (optionally) writes CSVs.

Dependencies:  pandas  pycountry
    >>  pip install pandas pycountry
"""

from pathlib import Path
import pandas as pd
import pycountry

# ───────────────────────────────────────────────────────────────────────────────
# 1.  Universal helper
# ───────────────────────────────────────────────────────────────────────────────

def tidy_governance_file(
    file_path: str | Path,
    sheet: str | int = "data",
    *,
    drop_empty_cols: bool = False,
    drop_empty_rows: bool = False,
    default_variable: str | None = None,
    default_scenario: str = "Baseline",
) -> pd.DataFrame:
    """Load → clean → reshape → return a fully de-NA’ed long DataFrame."""
    # ── load
    df = pd.read_excel(file_path, sheet_name=sheet)

    # ── header hygiene
    df.columns = (
        df.columns.astype(str)
                  .str.strip()
                  .str.replace(r"\s+", " ", regex=True)
    )

    # ── optional pre-drops
    if drop_empty_cols:
        df = df.dropna(axis=1, how="all")
    if drop_empty_rows:
        df = df.dropna(axis=0, how="all")

    # ── drop first "Model" col if present
    if df.columns[0].strip().lower() == "model":
        df = df.drop(columns=[df.columns[0]])

    # ── rename / guarantee key cols
    for col in df.columns:
        if col.lower() == "region":
            df = df.rename(columns={col: "Country"})
            break

    if "Scenario" not in df.columns:
        df["Scenario"] = default_scenario
    if "Variable" not in df.columns:
        if default_variable is None:
            raise ValueError(
                f"{file_path}: no 'Variable' column and no default_variable supplied"
            )
        df["Variable"] = default_variable
    if "Unit" in df.columns:
        df = df.drop(columns=["Unit"])

    # ── ISO-3 codes
    def to_iso(name: str) -> str | None:
        try:
            return pycountry.countries.lookup(name).alpha_3
        except LookupError:
            return None

    df["ISO"] = df["Country"].apply(to_iso)
    df = df.dropna(subset=["ISO"])              # keep only real countries

    # ── reshape to long
    year_cols = [c for c in df.columns if c.isdigit()]
    id_cols   = ["Scenario", "Country", "Variable", "ISO"]

    df_long = (
        df.melt(id_vars=id_cols, value_vars=year_cols,
                var_name="Year", value_name="Value")
          .assign(Year=lambda d: d["Year"].astype(int))
          .sort_values(id_cols + ["Year"])
          .dropna(subset=id_cols + ["Year", "Value"])   # remove ANY remaining NaNs
          .reset_index(drop=True)
    )

    return df_long


# ───────────────────────────────────────────────────────────────────────────────
# 2.  File paths  (adjust if your project lives elsewhere)
# ───────────────────────────────────────────────────────────────────────────────

base = Path(
    "/Users/valentinadlc/Library/Caches/JetBrains/PyCharm2025.1/"
    "demo/PyCharmLearningProject/data1/SSPS"
)

paths = {
    "governance":   base / "governance.xlsx",
    "urbanization": base / "urbanization.xlsx",
    "rule_law":     base / "rule_law.xlsx",
}

# ───────────────────────────────────────────────────────────────────────────────
# 3.  Build tidy master tables
# ───────────────────────────────────────────────────────────────────────────────

df_gov  = tidy_governance_file(paths["governance"],
                               default_variable="Governance")

df_urb  = tidy_governance_file(paths["urbanization"],
                               drop_empty_cols=True,
                               default_variable="Urbanization")

df_rule = tidy_governance_file(paths["rule_law"],
                               drop_empty_rows=True,
                               default_variable="Rule of Law")

# ───────────────────────────────────────────────────────────────────────────────
# 4.  Year-range splits
# ───────────────────────────────────────────────────────────────────────────────

# Governance splits (legacy request: 1996-2015 vs 2015-2099)
df_gov_1996_2015 = df_gov[df_gov["Year"].between(1996, 2015)].copy()
df_gov_2015_2099 = df_gov[df_gov["Year"].between(2015, 2099)].copy()
df_gov_1996_2024 = df_gov[df_gov["Year"].between(1996, 2024)].copy()

# Urbanization & Rule-of-Law splits (≤2024 vs ≥2025)
df_urb_pre2024   = df_urb[df_urb["Year"] <= 2024].copy()
df_urb_2025plus  = df_urb[df_urb["Year"] >= 2025].copy()

df_rule_pre2024  = df_rule[df_rule["Year"] <= 2024].copy()
df_rule_2025plus = df_rule[df_rule["Year"] >= 2025].copy()

# ───────────────────────────────────────────────────────────────────────────────
# 5.  (Optional) persist results as CSV
# ───────────────────────────────────────────────────────────────────────────────

# Uncomment the block below if you want the files on disk
df_gov_1996_2015.to_csv("governance_1996_2015.csv", index=False)
df_gov_2015_2099.to_csv("governance_2015_2099.csv", index=False)
df_gov_1996_2024.to_csv("governance_1996_2024.csv", index=False)
#
df_urb_pre2024.to_csv("urbanization_pre2024.csv", index=False)
df_urb_2025plus.to_csv("urbanization_2025plus.csv", index=False)
#
df_rule_pre2024.to_csv("rule_law_pre2024.csv", index=False)
df_rule_2025plus.to_csv("rule_law_2025plus.csv", index=False)

# Quick confirmation prints (can delete)
if __name__ == "__main__":
    print("Governance rows:",   len(df_gov))
    print("Urbanization rows:", len(df_urb))
    print("Rule-of-Law rows:",  len(df_rule))
