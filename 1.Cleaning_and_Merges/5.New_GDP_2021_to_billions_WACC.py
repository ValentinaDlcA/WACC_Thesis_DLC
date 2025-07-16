#DATA CLEANING WB GDP PPP CONSTANT 2021 TRANSFORMED TO BILLIONS TO MATCH WITH PROJECTED GDP PPP FROM NFGS
#CREATES FINAL DATASET CALLED "wacc_with_gdpppp2021.csv"
import pandas as pd
import pycountry
import re
import csv

path = "/Users/valentinadlc/Documents/MASTER/MASTER THESIS/WACC_Thesis_DLC/Data/data1_old/GDP/gdp_ppp_2021constant.csv"  # <-- set full path


# 1) Read file line-by-line, skipping metadata rows
# ------------------------------------------------------------------
rows = []
with open(path, encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        # Skip the first 4 metadata lines
        if row and row[0].startswith("Country Name"):
            header = row
            break
    # Read the remaining rows
    for row in reader:
        if row:                       # ignore blank lines
            rows.append(row)

# ------------------------------------------------------------------
# 2) Clean header: strip " [YRxxxx]" from year columns
# ------------------------------------------------------------------
new_header = []
for col in header:
    # e.g. "2021 [YR2021]" -> "2021"
    m = re.match(r"(\d{4})\s+\[YR\d{4}\]", col)
    new_header.append(m.group(1) if m else col.strip())

# ------------------------------------------------------------------
# 3) Build DataFrame
# ------------------------------------------------------------------
df = pd.DataFrame(rows, columns=new_header)

# ------------------------------------------------------------------
# 4) Basic column housekeeping
# ------------------------------------------------------------------
df = df.rename(columns={
    "Country Code": "ISO",
    "Indicator Name": "gdp_ppp_2021"
}).drop(columns=["Indicator Code"], errors="ignore")

# ------------------------------------------------------------------
# 5) Ensure valid ISO – patch special cases
# ------------------------------------------------------------------
manual_iso = {"Taiwan, China": "TWN", "Kosovo": "XKX"}
def iso_or_lookup(row):
    iso = row["ISO"]
    if iso and re.fullmatch(r"[A-Z]{3}", iso):
        return iso
    name = row["Country Name"].split(",")[0]  # strip suffix like ", China"
    return manual_iso.get(row["Country Name"]) or manual_iso.get(name) \
        or (lambda n: pycountry.countries.lookup(n).alpha_3
                     if pycountry.countries.get(name=n) else None)(name)

df["ISO"] = df.apply(iso_or_lookup, axis=1)
df = df[df["ISO"].notna()].copy()

# ------------------------------------------------------------------
# 6) Drop early-year columns (<2001)
# ------------------------------------------------------------------
cols_to_drop = [c for c in df.columns if c.isdigit() and int(c) < 2001]
df = df.drop(columns=cols_to_drop)

# ------------------------------------------------------------------
# 7) Drop rows with ANY NaN from 2008+
# ------------------------------------------------------------------
cols_2008_plus = [c for c in df.columns if c.isdigit() and int(c) >= 2008]
df = df.dropna(subset=cols_2008_plus, how="any")

# ------------------------------------------------------------------
# 8) Remove region / aggregate rows
# ------------------------------------------------------------------
region_pattern = re.compile(
    r"World|Europe|Asia|Africa|America|Caribbean|OECD|income|Middle East|"
    r"Euro area|Arab World|Least developed|IDA|IBRD|LDC|Small states|"
    r"Sub-Saharan|Pacific|G20|G7", re.IGNORECASE)
df = df[~df["Country Name"].str.contains(region_pattern, na=False)]

# ------------------------------------------------------------------
# 9) Melt to long format & convert to billions
gdp_ppp_long = df.melt(
    id_vars=["Country Name", "ISO", "gdp_ppp_2021"],
    var_name="Year",
    value_name="GDP_PPP"
).reset_index(drop=True)

# --- safe numeric conversion of Year -----------------------------------
gdp_ppp_long["Year"] = pd.to_numeric(gdp_ppp_long["Year"], errors="coerce")
gdp_ppp_long = gdp_ppp_long.dropna(subset=["Year"])
gdp_ppp_long["Year"] = gdp_ppp_long["Year"].astype(int)


gdp_ppp_long = gdp_ppp_long.dropna(subset=["GDP_PPP"])

# quick check
print(gdp_ppp_long.head())
print("Rows after cleaning:", len(gdp_ppp_long))
print("Distinct countries:", gdp_ppp_long["ISO"].nunique())
gdp_ppp_long.to_csv("gdp_ppp_2021_long.csv", index=False)

#NEW MERGE WITH WACC

# 1️⃣  Load the cleaned GDP-PPP long file you just created
#     (if it's still in memory, skip this read and reuse the DataFrame)
gdp_ppp_long = pd.read_csv("../Data/data1_old/GDP/gdp_ppp_2021_long.csv")   # update path if saved elsewhere

# 2️⃣  Load the WACC dataset
wacc_path = "/Users/valentinadlc/Documents/MASTER/MASTER THESIS/WACC_Thesis_DLC/WACC_Thesis_old/merged_with_wacc_updated.csv"
wacc_df = pd.read_csv(wacc_path)

# 3️⃣  Align the Year column type
gdp_ppp_long["Year"] = gdp_ppp_long["Year"].astype(int)
wacc_df["year"] = wacc_df["year"].astype(int)          # ensure correct column name
wacc_df = wacc_df.rename(columns={"year": "Year"})     # rename to match

# 4️⃣  Merge WACC (left) with GDP-PPP (right)
wacc_gdp_merged = pd.merge(
    wacc_df,
    gdp_ppp_long[["ISO", "Year", "GDP_PPP"]],
    on=["ISO", "Year"],
    how="left"
)

# 5️⃣  Quick summary
print("Merged rows:", len(wacc_gdp_merged))
print("Rows missing GDP_PPP:", wacc_gdp_merged['GDP_PPP'].isna().sum())
print(wacc_gdp_merged.head())

# 6️⃣  Save if desired
wacc_gdp_merged.to_csv("wacc_with_gdpppp2021.csv", index=False)
print("✅ Saved merged file: wacc_with_gdpppp2021.csv")

print("All columns in merged DF:")
print(wacc_gdp_merged.columns.tolist())