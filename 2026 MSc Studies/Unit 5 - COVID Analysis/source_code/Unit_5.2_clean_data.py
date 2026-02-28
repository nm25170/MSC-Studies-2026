# Unit 5.2 - Load and Clean COVID-19 Dataset (London)
# Libraries: Pandas, Matplotlib

import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

INPUT_FILE  = "london_covid_data.csv"
OUTPUT_FILE = "london_covid_clean.csv"

# ============================================================
# STEP 1 – LOAD
# ============================================================
print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

df_raw = pd.read_csv(INPUT_FILE)

print(f"Shape           : {df_raw.shape}  (rows, columns)")
print(f"Columns         : {list(df_raw.columns)}")
print(f"\nRaw dtypes:\n{df_raw.dtypes}")
print(f"\nFirst 3 rows:\n{df_raw.head(3).to_string(index=False)}")

# ============================================================
# STEP 2 – INSPECT: missing values & duplicates
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Inspecting data quality")
print("=" * 60)

print(f"\nMissing values per column (before cleaning):")
print(df_raw.isnull().sum().to_string())

dup_count = df_raw.duplicated().sum()
print(f"\nDuplicate rows  : {dup_count}")

# ============================================================
# STEP 3 – CLEAN
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Cleaning")
print("=" * 60)

df = df_raw.copy()

# --- 3a. Parse date ---
df["date"] = pd.to_datetime(df["date"])
print("  [OK] 'date' converted to datetime")

# --- 3b. Sort by date ---
df = df.sort_values("date").reset_index(drop=True)
print("  [OK] Sorted chronologically")

# --- 3c. Rename columns to readable snake_case ---
df.rename(columns={
    "areaName"                       : "area_name",
    "areaCode"                       : "area_code",
    "newCasesByPublishDate"          : "new_cases",
    "cumCasesByPublishDate"          : "cum_cases",
    "newDeaths28DaysByPublishDate"   : "new_deaths",
    "cumDeaths28DaysByPublishDate"   : "cum_deaths",
    "newAdmissions"                  : "new_admissions",
    "cumAdmissions"                  : "cum_admissions",
    "hospitalCases"                  : "hospital_cases",
}, inplace=True)
print("  [OK] Columns renamed to snake_case")

# --- 3d. Drop duplicate rows (if any) ---
before = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  [OK] Duplicates removed: {before - len(df)} rows dropped")

# --- 3e. Fill missing numeric values ---
# Early pandemic rows have NaN for deaths/admissions/hospital — fill with 0
numeric_cols = [
    "new_cases", "cum_cases",
    "new_deaths", "cum_deaths",
    "new_admissions", "cum_admissions",
    "hospital_cases"
]
missing_before = df[numeric_cols].isnull().sum().sum()
df[numeric_cols] = df[numeric_cols].fillna(0)
print(f"  [OK] {missing_before} missing numeric values filled with 0")

# --- 3f. Convert float columns to int (no decimals needed) ---
df[numeric_cols] = df[numeric_cols].astype(int)
print("  [OK] Numeric columns converted from float to int")

# --- 3g. Validate: no negative values in metric columns ---
neg_mask = (df[numeric_cols] < 0).any(axis=1)
neg_count = neg_mask.sum()
if neg_count > 0:
    print(f"  [WARN] {neg_count} rows with negative values found — setting to 0")
    df[numeric_cols] = df[numeric_cols].clip(lower=0)
else:
    print("  [OK] No negative values detected")

# --- 3h. Add derived columns ---
df["case_fatality_rate_pct"] = (
    (df["cum_deaths"] / df["cum_cases"].replace(0, pd.NA)) * 100
).round(2).fillna(0)

df["hospitalisation_rate_pct"] = (
    (df["cum_admissions"] / df["cum_cases"].replace(0, pd.NA)) * 100
).round(2).fillna(0)

df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month
print("  [OK] Derived columns added: case_fatality_rate_pct, hospitalisation_rate_pct, year, month")

# ============================================================
# STEP 4 – SUMMARY: before vs after
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Cleaned dataset summary")
print("=" * 60)
print(f"Shape           : {df.shape}")
print(f"Date range      : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Area            : {df['area_name'].iloc[0]}  ({df['area_code'].iloc[0]})")
print(f"\nMissing values after cleaning:\n{df.isnull().sum().to_string()}")
print(f"\nData types:\n{df.dtypes.to_string()}")
print(f"\nCleaned data (first 5 rows):\n{df.head().to_string(index=False)}")
print(f"\nDescriptive statistics:\n{df[numeric_cols].describe().to_string()}")

# ============================================================
# STEP 5 – SAVE cleaned CSV
# ============================================================
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n[SAVED] Cleaned dataset → {os.path.abspath(OUTPUT_FILE)}")

# ============================================================
# STEP 6 – VISUALISE data quality (Matplotlib)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Generating data-quality charts")
print("=" * 60)

fig = plt.figure(figsize=(16, 14))
fig.suptitle("London COVID-19 Dataset — Data Cleaning Overview", fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ----- Chart 1: Missing values heatmap (raw vs clean) -----
ax1 = fig.add_subplot(gs[0, :])
raw_missing = df_raw.isnull().sum()
clean_missing = pd.Series(0, index=raw_missing.index)   # all zeros after cleaning
x = range(len(raw_missing))
width = 0.35
ax1.bar([i - width/2 for i in x], raw_missing.values, width, label="Before Cleaning", color="#e74c3c", alpha=0.85)
ax1.bar([i + width/2 for i in x], clean_missing.values, width, label="After Cleaning",  color="#2ecc71", alpha=0.85)
ax1.set_xticks(list(x))
ax1.set_xticklabels(raw_missing.index, rotation=30, ha="right", fontsize=8)
ax1.set_ylabel("Missing Value Count")
ax1.set_title("Missing Values: Before vs After Cleaning")
ax1.legend()
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# ----- Chart 2: New cases over time -----
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(df["date"], df["new_cases"], color="#3498db", linewidth=1.5)
ax2.fill_between(df["date"], df["new_cases"], alpha=0.2, color="#3498db")
ax2.set_title("Weekly New Cases — London")
ax2.set_xlabel("Date")
ax2.set_ylabel("New Cases")
ax2.tick_params(axis="x", rotation=30)
ax2.grid(linestyle="--", alpha=0.4)

# ----- Chart 3: New deaths over time -----
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(df["date"], df["new_deaths"], color="#e74c3c", linewidth=1.5)
ax3.fill_between(df["date"], df["new_deaths"], alpha=0.2, color="#e74c3c")
ax3.set_title("Weekly New Deaths (28-day) — London")
ax3.set_xlabel("Date")
ax3.set_ylabel("New Deaths")
ax3.tick_params(axis="x", rotation=30)
ax3.grid(linestyle="--", alpha=0.4)

# ----- Chart 4: Case fatality rate over time -----
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(df["date"], df["case_fatality_rate_pct"], color="#9b59b6", linewidth=1.5)
ax4.set_title("Case Fatality Rate (%) Over Time — London")
ax4.set_xlabel("Date")
ax4.set_ylabel("CFR (%)")
ax4.tick_params(axis="x", rotation=30)
ax4.grid(linestyle="--", alpha=0.4)

# ----- Chart 5: Data type distribution (cleaned) -----
ax5 = fig.add_subplot(gs[2, 1])
dtype_counts = df.dtypes.astype(str).value_counts()
ax5.bar(dtype_counts.index, dtype_counts.values, color=["#f39c12", "#1abc9c", "#e74c3c"], alpha=0.85)
ax5.set_title("Column Data Types (Cleaned Dataset)")
ax5.set_xlabel("Data Type")
ax5.set_ylabel("Number of Columns")
ax5.grid(axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(dtype_counts.values):
    ax5.text(i, v + 0.05, str(v), ha="center", fontweight="bold")

chart_file = "london_covid_cleaning_overview.png"
plt.savefig(chart_file, dpi=150, bbox_inches="tight")
plt.close()
print(f"[SAVED] Chart → {os.path.abspath(chart_file)}")
print("\nAll cleaning steps completed successfully.")
