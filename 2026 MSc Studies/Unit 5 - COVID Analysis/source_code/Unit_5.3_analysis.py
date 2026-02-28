# Unit 5.3 - Basic Analysis of London COVID-19 Dataset
# Analysis: Descriptive statistics, trends, and correlations
# Libraries: Pandas, Matplotlib

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import os

INPUT_FILE = "london_covid_clean.csv"

METRIC_COLS = [
    "new_cases", "new_deaths", "new_admissions",
    "hospital_cases", "case_fatality_rate_pct"
]

# ============================================================
# LOAD
# ============================================================
df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"Loaded {len(df)} records  |  {df['date'].min().date()} → {df['date'].max().date()}\n")

# ============================================================
# SECTION 1 – DESCRIPTIVE STATISTICS
# ============================================================
print("=" * 65)
print("SECTION 1: Descriptive Statistics")
print("=" * 65)

stats = df[METRIC_COLS].agg(["mean", "median", "std", "min", "max", "skew"])
stats.loc["mode"] = df[METRIC_COLS].mode().iloc[0]

pd.set_option("display.float_format", "{:,.2f}".format)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 120)
print(stats.T.to_string())

# Per-year breakdown
print("\n--- Mean values by year ---")
print(df.groupby("year")[METRIC_COLS].mean().round(1).to_string())

# Per-wave peak (identify top 3 weeks by new cases)
print("\n--- Top 5 peak weeks by new cases ---")
top5 = df.nlargest(5, "new_cases")[["date", "new_cases", "new_deaths", "hospital_cases", "case_fatality_rate_pct"]]
print(top5.to_string(index=False))

# ============================================================
# SECTION 2 – TREND ANALYSIS
# ============================================================
print("\n" + "=" * 65)
print("SECTION 2: Trend Analysis")
print("=" * 65)

# 4-week rolling average
df["cases_4wk_avg"]      = df["new_cases"].rolling(window=4, min_periods=1).mean().round(0)
df["deaths_4wk_avg"]     = df["new_deaths"].rolling(window=4, min_periods=1).mean().round(0)
df["admissions_4wk_avg"] = df["new_admissions"].rolling(window=4, min_periods=1).mean().round(0)

# Month-over-month % change in new cases
monthly = (
    df.groupby(["year", "month"])["new_cases"]
    .sum()
    .reset_index(name="monthly_cases")
)
monthly["mom_change_pct"] = monthly["monthly_cases"].pct_change().mul(100).round(1)

print("\n--- Monthly new cases & month-over-month change ---")
print(monthly.to_string(index=False))

# Overall trend direction per year
print("\n--- Year-on-year total cases ---")
yearly = df.groupby("year")["new_cases"].sum()
print(yearly.to_string())

# Wave identification (periods where new_cases > 75th percentile)
threshold = df["new_cases"].quantile(0.75)
df["high_transmission"] = df["new_cases"] > threshold
waves = df[df["high_transmission"]][["date", "new_cases", "new_deaths"]]
print(f"\n--- High-transmission weeks (above 75th percentile: {threshold:,.0f} cases) ---")
print(f"Number of high-transmission weeks : {len(waves)}")
print(waves.to_string(index=False))

# ============================================================
# SECTION 3 – CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 65)
print("SECTION 3: Correlation Analysis")
print("=" * 65)

corr_cols = ["new_cases", "new_deaths", "new_admissions", "hospital_cases", "case_fatality_rate_pct"]
corr_matrix = df[corr_cols].corr().round(3)

print("\nPearson Correlation Matrix:")
print(corr_matrix.to_string())

# Lag correlation: deaths lag cases by ~2 weeks
df["new_cases_lag2"] = df["new_cases"].shift(2)
lag_corr = df[["new_cases_lag2", "new_deaths"]].dropna().corr().iloc[0, 1]
print(f"\nLag correlation (deaths vs cases 2 weeks prior): r = {lag_corr:.3f}")

df["new_cases_lag3"] = df["new_cases"].shift(3)
lag_corr3 = df[["new_cases_lag3", "new_deaths"]].dropna().corr().iloc[0, 1]
print(f"Lag correlation (deaths vs cases 3 weeks prior): r = {lag_corr3:.3f}")

# ============================================================
# SECTION 4 – CHARTS
# ============================================================
print("\n" + "=" * 65)
print("SECTION 4: Generating analysis charts")
print("=" * 65)

BLUE   = "#2980b9"
RED    = "#e74c3c"
GREEN  = "#27ae60"
PURPLE = "#8e44ad"
ORANGE = "#e67e22"
GREY   = "#95a5a6"

fig = plt.figure(figsize=(18, 22))
fig.suptitle("London COVID-19 — Basic Analysis\n(Mar 2020 – Apr 2022)",
             fontsize=16, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)


# ── Chart 1: New cases with 4-week rolling average ───────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.bar(df["date"], df["new_cases"], width=6, color=BLUE, alpha=0.4, label="Weekly new cases")
ax1.plot(df["date"], df["cases_4wk_avg"], color=BLUE, linewidth=2.2, label="4-week rolling avg")
ax1.axhline(df["new_cases"].mean(), color=GREY, linestyle="--", linewidth=1.2,
            label=f"Mean: {df['new_cases'].mean():,.0f}")
ax1.axhline(df["new_cases"].median(), color=GREEN, linestyle=":", linewidth=1.5,
            label=f"Median: {df['new_cases'].median():,.0f}")
ax1.set_title("New Cases Over Time — with Mean, Median & Rolling Average", fontsize=12)
ax1.set_ylabel("New Cases")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax1.legend(fontsize=9)
ax1.grid(linestyle="--", alpha=0.35)
ax1.tick_params(axis="x", rotation=30)


# ── Chart 2: New deaths with 4-week rolling average ──────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.bar(df["date"], df["new_deaths"], width=6, color=RED, alpha=0.4, label="Weekly new deaths")
ax2.plot(df["date"], df["deaths_4wk_avg"], color=RED, linewidth=2, label="4-week rolling avg")
ax2.axhline(df["new_deaths"].mean(),   color=GREY,  linestyle="--", linewidth=1.2,
            label=f"Mean: {df['new_deaths'].mean():.0f}")
ax2.axhline(df["new_deaths"].median(), color=GREEN, linestyle=":",  linewidth=1.5,
            label=f"Median: {df['new_deaths'].median():.0f}")
ax2.set_title("New Deaths (28-day) Over Time", fontsize=11)
ax2.set_ylabel("New Deaths")
ax2.legend(fontsize=8)
ax2.grid(linestyle="--", alpha=0.35)
ax2.tick_params(axis="x", rotation=30)


# ── Chart 3: Hospital cases over time ────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.fill_between(df["date"], df["hospital_cases"], alpha=0.35, color=ORANGE)
ax3.plot(df["date"], df["hospital_cases"], color=ORANGE, linewidth=1.8)
ax3.axhline(df["hospital_cases"].mean(),   color=GREY,  linestyle="--", linewidth=1.2,
            label=f"Mean: {df['hospital_cases'].mean():,.0f}")
ax3.axhline(df["hospital_cases"].median(), color=GREEN, linestyle=":",  linewidth=1.5,
            label=f"Median: {df['hospital_cases'].median():,.0f}")
ax3.set_title("Hospital Cases Over Time", fontsize=11)
ax3.set_ylabel("Hospital Cases")
ax3.legend(fontsize=8)
ax3.grid(linestyle="--", alpha=0.35)
ax3.tick_params(axis="x", rotation=30)


# ── Chart 4: Correlation heatmap ──────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
cmap_data = corr_matrix.values
im = ax4.imshow(cmap_data, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax4.set_xticks(range(len(corr_cols)))
ax4.set_yticks(range(len(corr_cols)))
short_labels = ["New\nCases", "New\nDeaths", "Admissions", "Hospital\nCases", "CFR (%)"]
ax4.set_xticklabels(short_labels, fontsize=8)
ax4.set_yticklabels(short_labels, fontsize=8)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        val = corr_matrix.values[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax4.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                 fontweight="bold", color=color)
plt.colorbar(im, ax=ax4, shrink=0.85)
ax4.set_title("Correlation Heatmap", fontsize=11)


# ── Chart 5: Scatter – cases vs deaths (with lag) ────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
valid = df[["new_cases_lag2", "new_deaths"]].dropna()
ax5.scatter(valid["new_cases_lag2"], valid["new_deaths"],
            alpha=0.6, color=PURPLE, edgecolors="white", linewidths=0.4, s=55)
# trend line
import numpy as np
z = np.polyfit(valid["new_cases_lag2"], valid["new_deaths"], 1)
p = np.poly1d(z)
x_line = np.linspace(valid["new_cases_lag2"].min(), valid["new_cases_lag2"].max(), 200)
ax5.plot(x_line, p(x_line), color=RED, linewidth=1.8, linestyle="--",
         label=f"Trend  (r = {lag_corr:.3f})")
ax5.set_title("New Deaths vs New Cases (2-week lag)", fontsize=11)
ax5.set_xlabel("New Cases (2 weeks prior)")
ax5.set_ylabel("New Deaths")
ax5.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax5.legend(fontsize=9)
ax5.grid(linestyle="--", alpha=0.35)


# ── Chart 6: Monthly total cases bar chart ───────────────────────────────────
ax6 = fig.add_subplot(gs[3, :])
monthly["label"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
colors = [BLUE if y == 2020 else GREEN if y == 2021 else ORANGE
          for y in monthly["year"]]
bars = ax6.bar(range(len(monthly)), monthly["monthly_cases"], color=colors, alpha=0.85)
ax6.set_xticks(range(len(monthly)))
ax6.set_xticklabels(monthly["label"], rotation=60, ha="right", fontsize=7.5)
ax6.set_title("Monthly Total New Cases — London  (Blue=2020, Green=2021, Orange=2022)", fontsize=11)
ax6.set_ylabel("Total Monthly Cases")
ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax6.grid(axis="y", linestyle="--", alpha=0.35)

# annotate MoM % change on selected bars
for i, row in monthly.iterrows():
    if pd.notna(row["mom_change_pct"]) and abs(row["mom_change_pct"]) > 50:
        ax6.text(i, monthly.loc[i, "monthly_cases"] + 5000,
                 f"{row['mom_change_pct']:+.0f}%",
                 ha="center", fontsize=6.5, color="black", fontweight="bold")

chart_file = "london_covid_analysis.png"
plt.savefig(chart_file, dpi=150, bbox_inches="tight")
plt.close()
print(f"[SAVED] Analysis chart → {os.path.abspath(chart_file)}")
print("\nAnalysis complete.")
