# ============================================================
# Unit 5.3.1 - Simplified COVID-19 Analysis (London)
# Beginner friendly version of Unit_5.3_analysis.py
# Libraries: Pandas, Matplotlib
# ============================================================

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# STEP 1 - LOAD THE DATASET
# ============================================================

# Read the cleaned CSV file into a DataFrame
df = pd.read_csv("london_covid_clean.csv")

# Convert the date column so Python understands it as a date
df["date"] = pd.to_datetime(df["date"])

print("Dataset loaded!")
print("Number of rows:", len(df))
print("Columns:", list(df.columns))
print()

# ============================================================
# STEP 2 - MEAN AND MEDIAN
# ============================================================
# Mean  = the average value
# Median = the middle value when all values are sorted

print("--- MEAN (Average) ---")
print("New Cases:      ", round(df["new_cases"].mean(), 1))
print("New Deaths:     ", round(df["new_deaths"].mean(), 1))
print("New Admissions: ", round(df["new_admissions"].mean(), 1))
print("Hospital Cases: ", round(df["hospital_cases"].mean(), 1))
print()

print("--- MEDIAN (Middle Value) ---")
print("New Cases:      ", df["new_cases"].median())
print("New Deaths:     ", df["new_deaths"].median())
print("New Admissions: ", df["new_admissions"].median())
print("Hospital Cases: ", df["hospital_cases"].median())
print()

print("--- MAX (Highest Value) ---")
print("New Cases:      ", df["new_cases"].max())
print("New Deaths:     ", df["new_deaths"].max())
print("Hospital Cases: ", df["hospital_cases"].max())
print()

# ============================================================
# STEP 3 - TREND ANALYSIS
# ============================================================
# A rolling average smooths out the data so we can see
# the overall direction (trend) more clearly.
# We use a 4-week window (4 rows at a time).

df["cases_rolling"]      = df["new_cases"].rolling(window=4).mean()
df["deaths_rolling"]     = df["new_deaths"].rolling(window=4).mean()
df["admissions_rolling"] = df["new_admissions"].rolling(window=4).mean()

# Group data by year to see yearly totals
yearly = df.groupby("year")["new_cases"].sum()

print("--- YEARLY TOTAL CASES ---")
for year, total in yearly.items():
    print(f"  {year}: {total:,} cases")
print()

# Find the week with the most cases (the peak)
peak_row = df.loc[df["new_cases"].idxmax()]
print(f"--- PEAK WEEK ---")
print(f"  Date:      {peak_row['date'].date()}")
print(f"  New Cases: {peak_row['new_cases']:,}")
print(f"  New Deaths:{peak_row['new_deaths']}")
print()

# ============================================================
# STEP 4 - CORRELATION
# ============================================================
# Correlation tells us how strongly two things are related.
# A value close to 1  = strong positive link
# A value close to -1 = strong negative link
# A value close to 0  = little or no link

print("--- CORRELATION BETWEEN METRICS ---")
corr = df[["new_cases", "new_deaths", "new_admissions", "hospital_cases"]].corr()
print(corr.round(2))
print()

# Highlight the most important relationships
r1 = round(corr.loc["new_deaths", "hospital_cases"], 2)
r2 = round(corr.loc["new_cases",  "new_admissions"], 2)
r3 = round(corr.loc["new_deaths", "new_admissions"], 2)

print(f"  Deaths vs Hospital Cases:  r = {r1}  (very strong link)")
print(f"  Cases  vs Admissions:      r = {r2}  (moderate link)")
print(f"  Deaths vs Admissions:      r = {r3}  (strong link)")
print()

# ============================================================
# STEP 5 - CHARTS
# ============================================================

# ----- Chart 1: New Cases Over Time (Line Chart) -----
plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["new_cases"], color="steelblue", alpha=0.4, label="Weekly Cases")
plt.plot(df["date"], df["cases_rolling"], color="steelblue", linewidth=2.5, label="4-Week Average")
plt.axhline(df["new_cases"].mean(),   color="grey",  linestyle="--", label=f"Mean: {df['new_cases'].mean():,.0f}")
plt.axhline(df["new_cases"].median(), color="green", linestyle=":",  label=f"Median: {df['new_cases'].median():,.0f}")
plt.title("New COVID-19 Cases Over Time — London", fontsize=13, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("New Cases")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("simplified_chart1_cases_trend.png", dpi=150)
plt.close()
print("Chart 1 saved: Cases Trend")

# ----- Chart 2: New Deaths Over Time (Line Chart) -----
plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["new_deaths"], color="tomato", alpha=0.4, label="Weekly Deaths")
plt.plot(df["date"], df["deaths_rolling"], color="tomato", linewidth=2.5, label="4-Week Average")
plt.axhline(df["new_deaths"].mean(),   color="grey",  linestyle="--", label=f"Mean: {df['new_deaths'].mean():.0f}")
plt.axhline(df["new_deaths"].median(), color="green", linestyle=":",  label=f"Median: {df['new_deaths'].median():.0f}")
plt.title("New COVID-19 Deaths Over Time — London", fontsize=13, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("New Deaths")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("simplified_chart2_deaths_trend.png", dpi=150)
plt.close()
print("Chart 2 saved: Deaths Trend")

# ----- Chart 3: Mean vs Median Bar Chart -----
metrics = ["New Cases", "New Deaths", "New Admissions", "Hospital Cases"]
means   = [df["new_cases"].mean(), df["new_deaths"].mean(),
           df["new_admissions"].mean(), df["hospital_cases"].mean()]
medians = [df["new_cases"].median(), df["new_deaths"].median(),
           df["new_admissions"].median(), df["hospital_cases"].median()]

x = [0, 1, 2, 3]

plt.figure(figsize=(11, 6))
plt.bar([i - 0.2 for i in x], means,   width=0.4, color="steelblue", label="Mean",   alpha=0.85)
plt.bar([i + 0.2 for i in x], medians, width=0.4, color="mediumseagreen", label="Median", alpha=0.85)
plt.xticks(x, metrics, fontsize=11)
plt.ylabel("Value")
plt.title("Mean vs Median for Key COVID-19 Metrics — London", fontsize=13, fontweight="bold")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("simplified_chart3_mean_median.png", dpi=150)
plt.close()
print("Chart 3 saved: Mean vs Median")

# ----- Chart 4: Yearly Total Cases Bar Chart -----
plt.figure(figsize=(7, 5))
years  = list(yearly.index)
totals = list(yearly.values)
colors = ["steelblue", "mediumseagreen", "darkorange"]
bars = plt.bar(years, totals, color=colors, alpha=0.85, width=0.5)
for bar, val in zip(bars, totals):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10000,
             f"{val:,}", ha="center", fontsize=10, fontweight="bold")
plt.title("Total New Cases per Year — London", fontsize=13, fontweight="bold")
plt.xlabel("Year")
plt.ylabel("Total Cases")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("simplified_chart4_yearly_cases.png", dpi=150)
plt.close()
print("Chart 4 saved: Yearly Cases")

# ----- Chart 5: Correlation Bar Chart -----
pair_names = [
    "Deaths vs\nHospital Cases",
    "Deaths vs\nAdmissions",
    "Cases vs\nAdmissions",
    "Cases vs\nDeaths",
]
pair_values = [r1, r3, r2,
               round(corr.loc["new_cases", "new_deaths"], 2)]
bar_colors = ["mediumseagreen" if v >= 0 else "tomato" for v in pair_values]

plt.figure(figsize=(9, 5))
bars = plt.bar(pair_names, pair_values, color=bar_colors, alpha=0.85)
plt.axhline(0,   color="black", linewidth=0.8)
plt.axhline(0.7, color="green",  linestyle="--", linewidth=1, label="Strong (r = 0.7)")
plt.axhline(0.4, color="orange", linestyle=":",  linewidth=1, label="Moderate (r = 0.4)")
for bar, val in zip(bars, pair_values):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.02,
             f"r = {val}", ha="center", fontsize=10, fontweight="bold")
plt.ylim(0, 1.1)
plt.ylabel("Correlation (r)")
plt.title("Correlation Between COVID-19 Metrics — London", fontsize=13, fontweight="bold")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("simplified_chart5_correlation.png", dpi=150)
plt.close()
print("Chart 5 saved: Correlation")

print()
print("All done! 5 charts have been saved.")
