# Unit 5.4 - Visualisations: Key Insights from London COVID-19 Analysis
# Source data: Unit_5.3_analysis.py  |  Clean data: london_covid_clean.csv
# Charts: Bar graphs & Line charts for Mean/Median, Trends, Correlations

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import os

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"  : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.titlesize"    : 13,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 11,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "figure.facecolor"  : "#f8f9fa",
    "axes.facecolor"    : "#ffffff",
})

BLUE   = "#2980b9"
RED    = "#e74c3c"
GREEN  = "#27ae60"
PURPLE = "#8e44ad"
ORANGE = "#e67e22"
TEAL   = "#16a085"
GREY   = "#7f8c8d"
LGREY  = "#bdc3c7"

SAVED = []

def fmt_k(x, _):
    """Format axis tick as comma-separated integer."""
    return f"{int(x):,}"

def save(fig, filename):
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    SAVED.append(filename)
    print(f"  [SAVED] {filename}")

# ── Load & prepare (mirrors Unit_5.3_analysis.py) ────────────────────────────
df = pd.read_csv("london_covid_clean.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

METRIC_COLS = ["new_cases", "new_deaths", "new_admissions",
               "hospital_cases", "case_fatality_rate_pct"]
LABELS      = ["New Cases", "New Deaths", "New Admissions",
               "Hospital Cases", "Case Fatality\nRate (%)"]
COLOURS     = [BLUE, RED, ORANGE, TEAL, PURPLE]

# Rolling averages
df["cases_4wk"]      = df["new_cases"].rolling(4, min_periods=1).mean()
df["deaths_4wk"]     = df["new_deaths"].rolling(4, min_periods=1).mean()
df["admissions_4wk"] = df["new_admissions"].rolling(4, min_periods=1).mean()
df["hospital_4wk"]   = df["hospital_cases"].rolling(4, min_periods=1).mean()
df["cfr_4wk"]        = df["case_fatality_rate_pct"].rolling(4, min_periods=1).mean()

# Monthly aggregates
monthly = (df.groupby(["year", "month"])["new_cases"]
           .sum().reset_index(name="monthly_cases"))
monthly["label"]          = (monthly["year"].astype(str) + "-"
                              + monthly["month"].astype(str).str.zfill(2))
monthly["mom_change_pct"] = monthly["monthly_cases"].pct_change().mul(100).round(1)

# Yearly totals
yearly_cases  = df.groupby("year")["new_cases"].sum()
yearly_deaths = df.groupby("year")["new_deaths"].sum()

# Correlation matrix & lag
corr_cols   = METRIC_COLS
corr_matrix = df[corr_cols].corr().round(3)
df["lag2"]  = df["new_cases"].shift(2)
lag_corr    = df[["lag2", "new_deaths"]].dropna().corr().iloc[0, 1]

# Wave threshold (75th percentile)
threshold = df["new_cases"].quantile(0.75)

print("Generating charts …\n")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Mean vs Median: side-by-side bar chart
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("London COVID-19 — Mean vs Median for Key Metrics", fontsize=14,
             fontweight="bold", y=1.02)

means   = [df[c].mean()   for c in METRIC_COLS]
medians = [df[c].median() for c in METRIC_COLS]
x       = np.arange(len(METRIC_COLS))
w       = 0.38

# Left: cases / deaths / admissions / hospital (large numbers)
ax = axes[0]
b1 = ax.bar(x[:4] - w/2, means[:4],   w, label="Mean",   color=BLUE,  alpha=0.85)
b2 = ax.bar(x[:4] + w/2, medians[:4], w, label="Median", color=GREEN, alpha=0.85)
ax.set_xticks(x[:4])
ax.set_xticklabels(LABELS[:4], fontsize=9)
ax.set_ylabel("Count")
ax.set_title("Mean vs Median — Cases, Deaths, Admissions, Hospital")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
            f"{bar.get_height():,.0f}", ha="center", fontsize=7.5, color=BLUE, fontweight="bold")
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
            f"{bar.get_height():,.0f}", ha="center", fontsize=7.5, color=GREEN, fontweight="bold")

# Right: CFR % (smaller scale)
ax2 = axes[1]
ax2.bar(0 - w/2, means[4],   w, label="Mean",   color=PURPLE, alpha=0.85)
ax2.bar(0 + w/2, medians[4], w, label="Median", color=TEAL,   alpha=0.85)
ax2.set_xticks([0])
ax2.set_xticklabels(["Case Fatality Rate (%)"])
ax2.set_ylabel("Percentage (%)")
ax2.set_title("Mean vs Median — Case Fatality Rate (%)")
ax2.legend()
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.text(-w/2, means[4]   * 1.03, f"{means[4]:.2f}%",   ha="center", fontsize=11, color=PURPLE, fontweight="bold")
ax2.text( w/2, medians[4] * 1.03, f"{medians[4]:.2f}%", ha="center", fontsize=11, color=TEAL,   fontweight="bold")

plt.tight_layout()
save(fig, "chart_1_mean_vs_median.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Mean by Year: grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════════
yearly_means = df.groupby("year")[METRIC_COLS[:4]].mean()   # exclude CFR (different scale)
years        = yearly_means.index.tolist()
n_metrics    = 4
n_years      = len(years)
x            = np.arange(n_metrics)
bar_w        = 0.25
year_colours = [BLUE, GREEN, ORANGE]

fig, ax = plt.subplots(figsize=(13, 6))
fig.suptitle("London COVID-19 — Mean Values by Year (Grouped Bar Chart)",
             fontsize=14, fontweight="bold")

for i, (year, col) in enumerate(zip(years, year_colours)):
    vals = yearly_means.loc[year].values
    bars = ax.bar(x + (i - 1) * bar_w, vals, bar_w,
                  label=str(year), color=col, alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.01,
                f"{h:,.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(LABELS[:4], fontsize=10)
ax.set_ylabel("Weekly Average Count")
ax.set_title("Average Weekly Cases, Deaths, Admissions & Hospital Cases — by Year")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
ax.legend(title="Year")
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
save(fig, "chart_2_mean_by_year.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Trend: New Cases line chart + rolling average
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(15, 6))
fig.suptitle("London COVID-19 — New Cases Trend Over Time", fontsize=14, fontweight="bold")

ax.bar(df["date"], df["new_cases"], width=6, color=BLUE, alpha=0.25, label="Weekly new cases")
ax.plot(df["date"], df["cases_4wk"], color=BLUE, linewidth=2.5, label="4-week rolling avg")
ax.axhline(df["new_cases"].mean(),   color=GREY,  linestyle="--", linewidth=1.5,
           label=f"Overall mean  ({df['new_cases'].mean():,.0f})")
ax.axhline(df["new_cases"].median(), color=GREEN, linestyle=":",  linewidth=1.8,
           label=f"Overall median ({df['new_cases'].median():,.0f})")
ax.axhline(threshold, color=RED, linestyle="-.", linewidth=1.3,
           label=f"75th pct threshold ({threshold:,.0f}) — high transmission")

# Annotate wave peaks
for _, row in df.nlargest(3, "new_cases").iterrows():
    ax.annotate(f"Peak\n{row['new_cases']:,}",
                xy=(row["date"], row["new_cases"]),
                xytext=(row["date"], row["new_cases"] + 15000),
                ha="center", fontsize=8, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

ax.set_xlabel("Date")
ax.set_ylabel("New Cases")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
ax.tick_params(axis="x", rotation=30)
ax.legend(loc="upper left")
ax.grid(linestyle="--", alpha=0.35)
plt.tight_layout()
save(fig, "chart_3_cases_trend.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Trend: New Deaths line chart + rolling average
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(15, 6))
fig.suptitle("London COVID-19 — New Deaths Trend Over Time (28-day basis)",
             fontsize=14, fontweight="bold")

ax.bar(df["date"], df["new_deaths"], width=6, color=RED, alpha=0.25, label="Weekly new deaths")
ax.plot(df["date"], df["deaths_4wk"], color=RED, linewidth=2.5, label="4-week rolling avg")
ax.axhline(df["new_deaths"].mean(),   color=GREY,  linestyle="--", linewidth=1.5,
           label=f"Mean   ({df['new_deaths'].mean():.0f})")
ax.axhline(df["new_deaths"].median(), color=GREEN, linestyle=":",  linewidth=1.8,
           label=f"Median ({df['new_deaths'].median():.0f})")

# Annotate peak death week
peak_death = df.loc[df["new_deaths"].idxmax()]
ax.annotate(f"Peak: {peak_death['new_deaths']:,}",
            xy=(peak_death["date"], peak_death["new_deaths"]),
            xytext=(peak_death["date"], peak_death["new_deaths"] + 50),
            ha="center", fontsize=9, color=RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

ax.set_xlabel("Date")
ax.set_ylabel("New Deaths")
ax.tick_params(axis="x", rotation=30)
ax.legend(loc="upper right")
ax.grid(linestyle="--", alpha=0.35)
plt.tight_layout()
save(fig, "chart_4_deaths_trend.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Trend: Hospital Cases + Admissions (dual-line chart)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(15, 6))
fig.suptitle("London COVID-19 — Hospital Cases & New Admissions Trends",
             fontsize=14, fontweight="bold")

ax1.plot(df["date"], df["hospital_4wk"], color=TEAL, linewidth=2.5,
         label="Hospital cases (4-wk avg)")
ax1.fill_between(df["date"], df["hospital_cases"], alpha=0.12, color=TEAL)
ax1.set_xlabel("Date")
ax1.set_ylabel("Hospital Cases", color=TEAL)
ax1.tick_params(axis="y", labelcolor=TEAL)
ax1.tick_params(axis="x", rotation=30)
ax1.axhline(df["hospital_cases"].mean(),   color=TEAL,  linestyle="--", linewidth=1.2, alpha=0.7,
            label=f"Hospital mean  ({df['hospital_cases'].mean():,.0f})")
ax1.axhline(df["hospital_cases"].median(), color=TEAL,  linestyle=":",  linewidth=1.5, alpha=0.7,
            label=f"Hospital median ({df['hospital_cases'].median():,.0f})")

ax2 = ax1.twinx()
ax2.bar(df["date"], df["new_admissions"], width=5, color=ORANGE, alpha=0.4,
        label="Weekly new admissions")
ax2.plot(df["date"], df["admissions_4wk"], color=ORANGE, linewidth=2,
         label="Admissions (4-wk avg)")
ax2.set_ylabel("New Admissions", color=ORANGE)
ax2.tick_params(axis="y", labelcolor=ORANGE)
ax2.axhline(df["new_admissions"].mean(),   color=ORANGE, linestyle="--", linewidth=1.2, alpha=0.6,
            label=f"Admissions mean ({df['new_admissions'].mean():.0f})")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
ax1.grid(linestyle="--", alpha=0.3)
plt.tight_layout()
save(fig, "chart_5_hospital_admissions_trend.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Trend: Case Fatality Rate over time
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(15, 6))
fig.suptitle("London COVID-19 — Case Fatality Rate (%) Trend Over Time",
             fontsize=14, fontweight="bold")

ax.plot(df["date"], df["case_fatality_rate_pct"], color=PURPLE, linewidth=1.5,
        alpha=0.5, label="Weekly CFR (%)")
ax.plot(df["date"], df["cfr_4wk"], color=PURPLE, linewidth=2.5,
        label="4-week rolling avg")
ax.fill_between(df["date"], df["cfr_4wk"], alpha=0.15, color=PURPLE)
ax.axhline(df["case_fatality_rate_pct"].mean(),   color=GREY,  linestyle="--", linewidth=1.5,
           label=f"Mean   ({df['case_fatality_rate_pct'].mean():.2f}%)")
ax.axhline(df["case_fatality_rate_pct"].median(), color=GREEN, linestyle=":",  linewidth=1.8,
           label=f"Median ({df['case_fatality_rate_pct'].median():.2f}%)")

# Shade pre- vs post-vaccination
vax_date = pd.Timestamp("2021-01-01")
ax.axvspan(df["date"].min(), vax_date, alpha=0.07, color=RED,  label="Pre-vaccination era")
ax.axvspan(vax_date, df["date"].max(), alpha=0.07, color=GREEN, label="Vaccination rollout era")
ax.axvline(vax_date, color=RED, linestyle="-", linewidth=1.2, alpha=0.5)
ax.text(vax_date, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 18,
        "Vaccines\nrolled out", fontsize=8, color=RED, ha="left")

ax.set_xlabel("Date")
ax.set_ylabel("Case Fatality Rate (%)")
ax.tick_params(axis="x", rotation=30)
ax.legend(loc="upper right", fontsize=8)
ax.grid(linestyle="--", alpha=0.35)
plt.tight_layout()
save(fig, "chart_6_cfr_trend.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 7 — Monthly Cases: bar chart with MoM % change line overlay
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(16, 6))
fig.suptitle("London COVID-19 — Monthly New Cases & Month-on-Month Change (%)",
             fontsize=14, fontweight="bold")

yr_cols = {2020: BLUE, 2021: GREEN, 2022: ORANGE}
bar_colours = [yr_cols[y] for y in monthly["year"]]
xpos = range(len(monthly))

ax1.bar(xpos, monthly["monthly_cases"], color=bar_colours, alpha=0.82)
ax1.set_xticks(list(xpos))
ax1.set_xticklabels(monthly["label"], rotation=60, ha="right", fontsize=8)
ax1.set_ylabel("Total Monthly Cases")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
ax1.grid(axis="y", linestyle="--", alpha=0.35)

# MoM % change overlay on secondary axis
ax2 = ax1.twinx()
ax2.plot(list(xpos), monthly["mom_change_pct"], color=RED, linewidth=2,
         marker="o", markersize=5, label="MoM change (%)", zorder=5)
ax2.axhline(0, color=GREY, linestyle="--", linewidth=0.9)
ax2.set_ylabel("Month-on-Month Change (%)", color=RED)
ax2.tick_params(axis="y", labelcolor=RED)

# Legend patches
patches = [mpatches.Patch(color=BLUE,   label="2020"),
           mpatches.Patch(color=GREEN,  label="2021"),
           mpatches.Patch(color=ORANGE, label="2022"),
           plt.Line2D([0], [0], color=RED, linewidth=2, marker="o", label="MoM Change (%)")]
ax1.legend(handles=patches, loc="upper left", fontsize=9)
plt.tight_layout()
save(fig, "chart_7_monthly_cases_bar.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 8 — Yearly totals: side-by-side bar chart (cases vs deaths)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("London COVID-19 — Year-on-Year Totals", fontsize=14, fontweight="bold")

years   = yearly_cases.index.tolist()
yc_vals = yearly_cases.values
yd_vals = yearly_deaths.values
yc_cols = [BLUE, GREEN, ORANGE]

ax = axes[0]
bars = ax.bar(years, yc_vals, color=yc_cols, alpha=0.85, width=0.5)
ax.set_title("Total New Cases by Year")
ax.set_ylabel("Total Cases")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
ax.grid(axis="y", linestyle="--", alpha=0.4)
for bar, val in zip(bars, yc_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
            f"{val:,}", ha="center", fontsize=10, fontweight="bold")

ax = axes[1]
bars = ax.bar(years, yd_vals, color=yc_cols, alpha=0.85, width=0.5)
ax.set_title("Total New Deaths by Year")
ax.set_ylabel("Total Deaths")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
ax.grid(axis="y", linestyle="--", alpha=0.4)
for bar, val in zip(bars, yd_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
            f"{val:,}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
save(fig, "chart_8_yearly_totals.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 9 — Correlation heatmap (standalone)
# ═══════════════════════════════════════════════════════════════════════════════
short = ["New Cases", "New Deaths", "Admissions", "Hospital Cases", "CFR (%)"]

fig, ax = plt.subplots(figsize=(8, 7))
fig.suptitle("London COVID-19 — Pearson Correlation Heatmap",
             fontsize=14, fontweight="bold")

im = ax.imshow(corr_matrix.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(short, rotation=30, ha="right", fontsize=10)
ax.set_yticklabels(short, fontsize=10)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Correlation (r)", fontsize=10)

for i in range(5):
    for j in range(5):
        val = corr_matrix.values[i, j]
        txt_col = "white" if abs(val) > 0.65 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=11, fontweight="bold", color=txt_col)

plt.tight_layout()
save(fig, "chart_9_correlation_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 10 — Correlation bar chart (r values for each pair)
# ═══════════════════════════════════════════════════════════════════════════════
pairs = [
    ("Deaths ↔ Hospital Cases",  corr_matrix.loc["new_deaths", "hospital_cases"]),
    ("Deaths ↔ Admissions",      corr_matrix.loc["new_deaths", "new_admissions"]),
    ("Cases ↔ Admissions",       corr_matrix.loc["new_cases",  "new_admissions"]),
    ("Hospital ↔ Admissions",    corr_matrix.loc["hospital_cases", "new_admissions"]),
    ("Cases ↔ Hospital Cases",   corr_matrix.loc["new_cases",  "hospital_cases"]),
    ("Cases ↔ Deaths",           corr_matrix.loc["new_cases",  "new_deaths"]),
    ("Cases ↔ CFR (%)",          corr_matrix.loc["new_cases",  "case_fatality_rate_pct"]),
    ("Deaths ↔ CFR (%)",         corr_matrix.loc["new_deaths", "case_fatality_rate_pct"]),
    ("Deaths vs Cases (lag 2w)", lag_corr),
]
pair_labels = [p[0] for p in pairs]
pair_vals   = [p[1] for p in pairs]
bar_cols    = [GREEN if v >= 0 else RED for v in pair_vals]

fig, ax = plt.subplots(figsize=(13, 6))
fig.suptitle("London COVID-19 — Pairwise Correlation Coefficients (r)",
             fontsize=14, fontweight="bold")

bars = ax.barh(pair_labels, pair_vals, color=bar_cols, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.9)
ax.axvline( 0.7, color=GREEN, linestyle="--", linewidth=1, alpha=0.6, label="|r| = 0.7 (strong)")
ax.axvline(-0.7, color=GREEN, linestyle="--", linewidth=1, alpha=0.6)
ax.axvline( 0.4, color=ORANGE, linestyle=":", linewidth=1, alpha=0.6, label="|r| = 0.4 (moderate)")
ax.axvline(-0.4, color=ORANGE, linestyle=":", linewidth=1, alpha=0.6)
ax.set_xlabel("Pearson r")
ax.set_title("Strength of Correlation Between COVID-19 Metrics")
ax.set_xlim(-1, 1)
ax.grid(axis="x", linestyle="--", alpha=0.35)
ax.legend(loc="lower right")
for bar, val in zip(bars, pair_vals):
    xpos = val + 0.02 if val >= 0 else val - 0.02
    ha   = "left" if val >= 0 else "right"
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f"r = {val:.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")

plt.tight_layout()
save(fig, "chart_10_correlation_bars.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 11 — Scatter: Cases vs Deaths (2-week lag) with trend line
# ═══════════════════════════════════════════════════════════════════════════════
valid = df[["lag2", "new_deaths"]].dropna()
z     = np.polyfit(valid["lag2"], valid["new_deaths"], 1)
p_fit = np.poly1d(z)
x_ln  = np.linspace(valid["lag2"].min(), valid["lag2"].max(), 300)

fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("London COVID-19 — Correlation Scatter: Deaths vs Cases (2-week lag)",
             fontsize=14, fontweight="bold")

sc = ax.scatter(valid["lag2"], valid["new_deaths"],
                alpha=0.65, color=PURPLE, edgecolors="white", linewidths=0.5, s=65,
                label="Weekly data points")
ax.plot(x_ln, p_fit(x_ln), color=RED, linewidth=2, linestyle="--",
        label=f"Linear trend  (r = {lag_corr:.3f})")
ax.axhline(df["new_deaths"].mean(),   color=GREY,  linestyle="--", linewidth=1,
           label=f"Deaths mean ({df['new_deaths'].mean():.0f})")
ax.axvline(df["new_cases"].mean(),    color=BLUE,  linestyle="--", linewidth=1,
           label=f"Cases mean ({df['new_cases'].mean():,.0f})")

ax.set_xlabel("New Cases (2 weeks prior)")
ax.set_ylabel("New Deaths")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
ax.legend()
ax.grid(linestyle="--", alpha=0.35)
plt.tight_layout()
save(fig, "chart_11_scatter_lag_correlation.png")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\nAll {len(SAVED)} charts saved:")
for s in SAVED:
    print(f"  → {os.path.abspath(s)}")
