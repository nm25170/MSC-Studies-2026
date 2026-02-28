# 2026 MSc Studies — University of Essex
## MSc Artificial Intelligence

This repository contains all coursework, data analysis projects, reports,
and source code produced during the MSc AI programme at the University of Essex.

---

## Repository Structure

```
2026 MSc Studies/
│
├── Unit 5 - COVID Analysis/
│   ├── source_code/          Python scripts (acquisition, cleaning, analysis, visualisation)
│   ├── charts/               All generated Matplotlib charts (PNG)
│   ├── data/                 Raw and cleaned CSV datasets
│   └── report/               Word document report and written reflections
│
└── Unit 6 - Housing Price Prediction/
    ├── source_code/          AI prediction model and report generation scripts
    ├── charts/               Matplotlib forecast and evaluation charts (PNG)
    ├── data/                 Dataset files (Land Registry / synthetic baseline)
    └── report/               Academic Word document with code, analysis, and references
```

---

## Unit 5 — COVID-19 Data Analysis (London Region)

| File | Description |
|---|---|
| `Unit_5_covid_analysis.py` | Dataset acquisition from UK Gov COVID-19 API |
| `Unit_5.2_clean_data.py` | Data cleaning and pre-processing pipeline |
| `Unit_5.3_analysis.py` | Full exploratory data analysis (mean, median, trends, correlations) |
| `Unit_5.3.1_simplified.py` | Beginner-friendly version of the analysis |
| `Unit_5.4_visualisations.py` | 11 dedicated bar and line chart visualisations |
| `export_to_word.py` | Generates the academic Word document submission |

### Dataset
- **Source**: UK Government COVID-19 Dashboard API (`coronavirus.data.gov.uk`)
- **Region**: London (area code: E12000007)
- **Period**: March 2020 – April 2022
- **Records**: 110 weekly observations

### Key Findings
- Three distinct transmission waves identified (Wave 1: Mar–May 2020, Wave 2/3: Oct 2020–Jul 2021, Omicron: Dec 2021–Feb 2022)
- Case counts rose from 135,245 (2020) → 1,700,200 (2021) → 906,000 (2022)
- Strong correlation between deaths and hospital cases (r = 0.91)
- Case Fatality Rate declined sharply from ~22% (early 2020) to <1% (2022) post-vaccination
- Peak week: 320,000 new cases (20 December 2021, Omicron surge)

---

---

## Unit 6 — AI Prediction Model: Littlehampton Housing Prices

| File | Description |
|---|---|
| `Unit_6_prediction_model.py` | Full AI prediction model — data acquisition, feature engineering, 4 ML models, forecast |
| `generate_unit6_report.py` | Generates the academic Word document submission |

### Dataset
- **Primary Source**: HM Land Registry Price Paid Data — SPARQL Linked Data API
  (`https://landregistry.data.gov.uk/landregistry/query`)
- **Secondary Source**: Land Registry annual CSV bulk download (S3)
- **Licence**: Open Government Licence v3.0
- **Location**: Littlehampton, West Sussex (Arun District)
- **Period**: Q1 2000 – Q4 2025 (quarterly aggregated)
- **Macroeconomic enrichment**: Bank of England base rate, ONS CPI inflation,
  ONS unemployment rate, ONS average annual income

### Models Trained
| Algorithm | Key Parameters |
|---|---|
| Linear Regression | Baseline ordinary least squares |
| Ridge Regression | L2 regularisation, α = 10.0 |
| Random Forest | 300 estimators, max depth 8 |
| Gradient Boosting | 300 estimators, lr = 0.05, depth 4 |

### Evaluation Metrics
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Coefficient of Determination (R²)
- 5-fold time-series cross-validation

### Forecast
Quarterly residential property price forecast for Littlehampton — **Q1 2026 to Q4 2030**,
based on Bank of England and OBR moderate economic scenario assumptions.

---

## Libraries Used
- `pandas` — data manipulation and analysis
- `numpy` — numerical computing
- `scikit-learn` — machine learning models and evaluation
- `matplotlib` — data visualisation and chart generation
- `python-docx` — academic report generation
- `requests` — API and dataset download

---

*University of Essex | MSc Artificial Intelligence | 2026*
