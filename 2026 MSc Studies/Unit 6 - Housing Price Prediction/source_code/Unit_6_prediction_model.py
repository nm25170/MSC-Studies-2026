# Unit 6 - AI Prediction Model: Littlehampton (UK) Housing Prices
# Dataset : HM Land Registry Price Paid Data (Open Government Licence v3.0)
#           SPARQL: https://landregistry.data.gov.uk/landregistry/query
#           CSV S3: http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com
# Goal    : Train ML models on real public transaction data,
#           evaluate accuracy, and forecast prices 2026–2030
# Models  : Linear Regression, Ridge, Random Forest, Gradient Boosting

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import os, time, warnings
import requests
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
np.random.seed(42)

BLUE   = "#2980b9"
RED    = "#e74c3c"
GREEN  = "#27ae60"
PURPLE = "#8e44ad"
ORANGE = "#e67e22"
GREY   = "#95a5a6"

# ============================================================
# MACROECONOMIC LOOKUP TABLE  (Bank of England / ONS figures)
# Used to enrich both real and synthetic datasets.
# ============================================================

_QUARTERS = pd.date_range("2000-01-01", "2025-10-01", freq="QS")

_BANK_RATE = [
    6.00,5.75,6.00,5.75, 5.75,5.25,5.00,4.00, 4.00,4.00,4.00,4.00,
    3.75,3.75,3.50,3.75, 4.00,4.25,4.50,4.75, 4.75,4.75,4.50,4.50,
    4.50,4.50,4.75,5.00, 5.25,5.50,5.75,5.50, 5.25,5.00,5.00,2.00,
    1.00,0.50,0.50,0.50, 0.50,0.50,0.50,0.50, 0.50,0.50,0.50,0.50,
    0.50,0.50,0.50,0.50, 0.50,0.50,0.50,0.50, 0.50,0.50,0.50,0.50,
    0.50,0.50,0.25,0.25, 0.25,0.25,0.25,0.50, 0.50,0.50,0.75,0.75,
    0.75,0.75,0.75,0.75, 0.75,0.10,0.10,0.10, 0.10,0.10,0.10,0.25,
    0.50,1.00,2.25,3.50, 4.00,4.50,5.25,5.25, 5.25,5.25,5.00,4.75,
    4.50,4.25,4.25,4.00,
]
_INFLATION = [
    0.8,0.9,1.1,1.0, 1.1,1.3,1.2,1.1, 1.3,1.4,1.5,1.7,
    1.4,1.5,1.4,1.3, 1.3,1.5,1.6,1.7, 1.6,1.9,2.3,2.2,
    1.9,2.0,2.5,2.7, 2.8,2.6,1.8,2.1, 2.2,3.3,4.8,3.1,
    3.0,2.1,1.6,2.9, 3.5,3.7,3.1,3.4, 4.0,4.5,4.7,4.2,
    3.6,3.0,2.4,2.7, 2.7,2.8,2.7,2.1, 1.9,1.7,1.5,0.9,
    0.3,0.1,0.0,0.2, 0.3,0.5,1.0,1.6, 1.8,2.7,2.8,3.0,
    3.0,2.4,2.4,2.3, 1.8,2.0,1.7,1.4, 1.8,0.9,0.5,0.3,
    0.4,2.0,3.1,5.1, 6.2,9.4,10.1,10.7, 10.4,8.7,6.8,4.6,
    3.4,2.0,2.2,2.5, 2.8,3.5,3.1,2.9,
]
_UNEMPLOYMENT = [
    5.6,5.5,5.4,5.3, 5.1,5.0,5.1,5.2, 5.1,5.2,5.3,5.1,
    5.0,5.0,5.0,4.9, 4.8,4.7,4.6,4.6, 4.7,4.8,4.8,5.0,
    5.3,5.5,5.4,5.5, 5.4,5.4,5.3,5.2, 5.2,5.3,5.8,6.3,
    6.7,7.6,7.9,7.8, 7.8,7.9,7.7,7.9, 7.8,7.7,8.1,8.4,
    8.3,8.2,7.9,7.7, 7.9,7.8,7.6,7.2, 6.9,6.6,6.2,5.8,
    5.7,5.5,5.4,5.1, 5.1,5.1,4.9,4.8, 4.7,4.6,4.3,4.3,
    4.3,4.2,4.0,4.0, 3.9,3.8,3.8,4.0, 4.0,3.9,4.5,5.0,
    5.1,4.9,4.6,4.2, 3.8,3.8,3.5,3.7, 3.7,4.0,4.3,4.2,
    4.4,4.4,4.3,4.4, 4.5,4.4,4.3,4.3,
]
_AVG_INCOME = [
    23400,23600,23800,24000, 24300,24600,24900,25200, 25500,25700,25900,26100,
    26400,26700,26900,27200, 27500,27800,28100,28400, 28700,29000,29300,29600,
    29900,30100,30400,30700, 31000,31400,31800,32100, 32400,32500,32300,31800,
    31500,31200,31100,31300, 31600,31800,32000,32100, 32000,31800,31700,31600,
    31800,32000,32200,32500, 32800,33100,33400,33800, 34200,34600,35100,35500,
    35900,36300,36700,37100, 37500,37800,38100,38400, 38700,39000,39200,39500,
    39800,40200,40600,41000, 41500,42000,42400,42800, 43000,42500,42200,42800,
    43500,44200,45000,45800, 46500,47200,47800,48500, 49500,50500,51200,51800,
    52300,52800,53200,53700, 54200,54700,55100,55500,
]

econ_df = pd.DataFrame({
    "date"        : _QUARTERS,
    "year"        : _QUARTERS.year,
    "quarter"     : _QUARTERS.quarter,
    "bank_rate"   : _BANK_RATE,
    "inflation"   : _INFLATION,
    "unemployment": _UNEMPLOYMENT,
    "avg_income"  : _AVG_INCOME,
})

# ============================================================
# SECTION 1 – DATA ACQUISITION
# ============================================================
print("=" * 65)
print("SECTION 1: Data Acquisition — HM Land Registry (Public Data)")
print("=" * 65)
print("Licence : Open Government Licence v3.0")
print("Source 1: https://landregistry.data.gov.uk/landregistry/query")
print("Source 2: http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com")
print()

SPARQL_URL  = "https://landregistry.data.gov.uk/landregistry/query"
PP_S3_BASE  = ("http://prod.publicdata.landregistry.gov.uk"
               ".s3-website-eu-west-1.amazonaws.com")

PT_URI_MAP  = {
    "detached"         : "Detached",
    "semi-detached"    : "Semi-Detached",
    "terraced"         : "Terraced",
    "flat-maisonette"  : "Flat",
    "otherPropertyType": "Other",
}
DUR_URI_MAP = {"freehold": "Freehold", "leasehold": "Leasehold"}

PP_CSV_COLS = [
    "uid","price","date","postcode","prop_type","new_build",
    "duration","paon","saon","street","locality","town",
    "district","county","ppd_cat","rec_status",
]
PT_CSV_MAP  = {"D":"Detached","S":"Semi-Detached","T":"Terraced",
               "F":"Flat","O":"Other"}
DUR_CSV_MAP = {"F":"Freehold","L":"Leasehold"}


def sparql_fetch_all():
    """Page through Land Registry SPARQL endpoint for Littlehampton."""
    rows, offset, LIMIT = [], 0, 5000
    while True:
        query = f"""
PREFIX lrppi:    <http://landregistry.data.gov.uk/def/ppi/>
PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>
PREFIX xsd:      <http://www.w3.org/2001/XMLSchema#>
SELECT ?price ?date ?propertyType ?newBuild ?duration
WHERE {{
  ?t lrppi:pricePaid       ?price  ;
     lrppi:transactionDate ?date   ;
     lrppi:propertyType    ?propertyType ;
     lrppi:newBuild        ?newBuild ;
     lrppi:estateType      ?duration ;
     lrppi:propertyAddress ?addr .
  ?addr lrcommon:town "LITTLEHAMPTON"^^xsd:string .
  FILTER(?date >= "2000-01-01"^^xsd:date)
}}
ORDER BY ?date LIMIT {LIMIT} OFFSET {offset}"""
        resp = requests.get(
            SPARQL_URL,
            params={"query": query, "output": "json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=90,
        )
        resp.raise_for_status()
        bindings = resp.json()["results"]["bindings"]
        if not bindings:
            break
        for b in bindings:
            pt  = PT_URI_MAP.get(b["propertyType"]["value"].split("/")[-1], "Other")
            dur = DUR_URI_MAP.get(b["duration"]["value"].split("/")[-1], "Unknown")
            rows.append({
                "price"        : float(b["price"]["value"]),
                "date"         : pd.to_datetime(b["date"]["value"]),
                "property_type": pt,
                "new_build"    : b["newBuild"]["value"].lower() == "true",
                "duration"     : dur,
            })
        print(f"  SPARQL: {len(rows):,} records (offset {offset})")
        if len(bindings) < LIMIT:
            break
        offset += LIMIT
        time.sleep(0.4)
    return pd.DataFrame(rows)


def csv_fetch_years(years):
    """Stream Land Registry annual CSVs from S3, filter for Littlehampton."""
    frames = []
    for yr in years:
        url = f"{PP_S3_BASE}/pp-{yr}.csv"
        print(f"  Streaming pp-{yr}.csv …")
        try:
            for chunk in pd.read_csv(
                url, header=None, names=PP_CSV_COLS,
                chunksize=100_000, parse_dates=["date"],
                dtype={"price": float}, low_memory=False,
            ):
                mask = chunk["town"].str.strip().str.upper() == "LITTLEHAMPTON"
                if mask.any():
                    frames.append(chunk.loc[mask])
        except Exception as exc:
            print(f"    {yr}: {exc}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["property_type"] = out["prop_type"].map(PT_CSV_MAP).fillna("Other")
    out["duration"]      = out["duration"].map(DUR_CSV_MAP).fillna("Unknown")
    out["new_build"]     = out["new_build"].str.strip().str.upper() == "Y"
    return out[["price", "date", "property_type", "new_build", "duration"]]


# ── Try sources in order ────────────────────────────────────────────────────
df_transactions = pd.DataFrame()
data_source     = ""

print("→ Attempting SPARQL API (Land Registry Linked Data)…")
try:
    df_transactions = sparql_fetch_all()
    data_source     = "HM Land Registry SPARQL API (Linked Data)"
    print(f"✓ SPARQL: {len(df_transactions):,} transactions\n")
except Exception as e:
    print(f"✗ SPARQL failed: {e}")
    print("\n→ Fallback: Land Registry annual CSV files (S3, 2010–2024)…")
    try:
        df_transactions = csv_fetch_years(range(2010, 2025))
        data_source     = "HM Land Registry Annual CSV (S3 bulk download)"
        print(f"✓ CSV: {len(df_transactions):,} transactions\n")
    except Exception as e2:
        print(f"✗ CSV also failed: {e2}")
        data_source = "Synthetic dataset (calibrated to ONS/Land Registry figures)"
        print(f"\n⚠  Both network sources unavailable.")
        print(f"   Using synthetic baseline dataset.\n")

print(f"Data source : {data_source}")

# ============================================================
# SECTION 2 – BUILD QUARTERLY AGGREGATE DATASET
# ============================================================
print("\n" + "=" * 65)
print("SECTION 2: Building Quarterly Dataset")
print("=" * 65)

if not df_transactions.empty:
    # ── Aggregate real transaction data to quarterly medians ─────────────
    df_transactions = df_transactions.dropna(subset=["price", "date"])
    df_transactions = df_transactions[df_transactions["price"] > 10_000]
    df_transactions["year"]    = df_transactions["date"].dt.year
    df_transactions["quarter"] = df_transactions["date"].dt.quarter

    qtr_agg = (
        df_transactions
        .groupby(["year", "quarter"])
        .agg(
            median_price   = ("price", "median"),
            mean_price     = ("price", "mean"),
            transaction_ct = ("price", "count"),
        )
        .reset_index()
    )
    qtr_agg["date"] = pd.to_datetime(
        qtr_agg["year"].astype(str) + "-" +
        ((qtr_agg["quarter"] - 1) * 3 + 1).astype(str).str.zfill(2) + "-01"
    )
    qtr_agg = qtr_agg.sort_values("date").reset_index(drop=True)
    # Merge economic indicators
    df = qtr_agg.merge(econ_df[["date","bank_rate","inflation",
                                 "unemployment","avg_income"]],
                       on="date", how="left")
    df["house_price"] = df["median_price"].round(0).astype(int)

    print(f"Real data: {len(df_transactions):,} individual transactions → "
          f"{len(df)} quarterly periods")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\nProperty type breakdown:")
    print(df_transactions["property_type"].value_counts().to_string())
    print(f"\nMedian price per property type:")
    print(df_transactions.groupby("property_type")["price"]
          .median().apply(lambda x: f"£{x:,.0f}").to_string())

else:
    # ── Synthetic quarterly dataset ───────────────────────────────────────
    _PRICES = np.array([
        88000,90000,91500,93000,    96000,99000,103000,108000,
        115000,122000,129000,136000, 143000,150000,157000,163000,
        167000,171000,174000,177000, 178000,179000,180000,181000,
        183000,186000,190000,195000, 199000,204000,208000,210000,
        207000,202000,194000,183000, 176000,172000,173000,177000,
        181000,185000,188000,186000, 183000,181000,180000,179000,
        179000,181000,183000,185000, 188000,192000,197000,203000,
        210000,218000,225000,231000, 237000,242000,247000,250000,
        253000,258000,263000,267000, 269000,271000,272000,271000,
        270000,268000,267000,265000, 265000,267000,270000,272000,
        273000,271000,282000,294000, 303000,313000,323000,332000,
        338000,342000,346000,343000, 336000,330000,326000,325000,
        327000,330000,334000,337000, 340000,344000,347000,350000,
    ], dtype=float)
    _PRICES += np.random.normal(0, 1500, len(_PRICES))

    df = econ_df.copy()
    df["house_price"]    = _PRICES.astype(int)
    df["median_price"]   = df["house_price"].astype(float)
    df["transaction_ct"] = np.random.randint(80, 200, len(df))

print(f"\nQuarterly records for modelling: {len(df)}")
print(f"Price range: £{df['house_price'].min():,} – £{df['house_price'].max():,}")

# ============================================================
# SECTION 3 – PROPERTY-LEVEL REGRESSION  (Model A)
#             Only runs when individual transaction data is available
# ============================================================
if not df_transactions.empty:
    print("\n" + "=" * 65)
    print("SECTION 3: Model A — Property Transaction Regression")
    print("=" * 65)

    # Build feature matrix for individual transactions
    tx = df_transactions.copy()
    tx = tx[tx["year"].between(2010, 2024)].copy()
    tx["time_idx"]  = (tx["year"] - 2010) * 4 + tx["quarter"] - 1
    tx["is_new"]    = tx["new_build"].astype(int)
    tx["is_fhd"]    = (tx["duration"] == "Freehold").astype(int)

    # Property type dummies (drop "Other" as baseline)
    for pt in ["Detached", "Semi-Detached", "Terraced", "Flat"]:
        tx[f"pt_{pt.replace('-','_').lower()}"] = (tx["property_type"] == pt).astype(int)

    # Merge economic features
    tx = tx.merge(
        econ_df[["year","quarter","bank_rate","inflation","unemployment","avg_income"]],
        on=["year","quarter"], how="left",
    ).dropna()

    TX_FEATURES = [
        "time_idx","is_new","is_fhd",
        "pt_detached","pt_semi_detached","pt_terraced","pt_flat",
        "bank_rate","inflation","unemployment","avg_income",
    ]
    TX_TARGET = "price"

    tx_split = int(len(tx) * 0.80)
    tx_train = tx.iloc[:tx_split]
    tx_test  = tx.iloc[tx_split:]

    print(f"Transaction records : {len(tx):,}  "
          f"(train {len(tx_train):,} / test {len(tx_test):,})")

    tx_models = {
        "Random Forest"    : RandomForestRegressor(n_estimators=200, max_depth=8,
                                                    min_samples_leaf=5, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                        learning_rate=0.05, random_state=42),
    }
    print(f"\n{'Model':<22}  {'MAE':>10}  {'RMSE':>10}  {'MAPE':>8}  {'R²':>7}")
    print("-" * 65)
    for name, model in tx_models.items():
        model.fit(tx_train[TX_FEATURES], tx_train[TX_TARGET])
        pred = np.clip(model.predict(tx_test[TX_FEATURES]), 0, None)
        mae  = mean_absolute_error(tx_test[TX_TARGET], pred)
        rmse = np.sqrt(mean_squared_error(tx_test[TX_TARGET], pred))
        mape = np.mean(np.abs((tx_test[TX_TARGET] - pred) / tx_test[TX_TARGET])) * 100
        r2   = r2_score(tx_test[TX_TARGET], pred)
        print(f"  {name:<22}  £{mae:>8,.0f}  £{rmse:>8,.0f}  {mape:>6.1f}%  {r2:>6.3f}")

# ============================================================
# SECTION 4 – FEATURE ENGINEERING  (Quarterly time-series)
# ============================================================
print("\n" + "=" * 65)
print("SECTION 4: Feature Engineering (Quarterly Time-Series)")
print("=" * 65)

df["time_idx"]        = np.arange(len(df))
df["time_idx_sq"]     = df["time_idx"] ** 2
df["price_lag1"]      = df["house_price"].shift(1)
df["price_lag2"]      = df["house_price"].shift(2)
df["price_lag4"]      = df["house_price"].shift(4)
df["price_roll4"]     = df["house_price"].rolling(4, min_periods=1).mean()
df["price_roll8"]     = df["house_price"].rolling(8, min_periods=1).mean()
df["price_yoy_pct"]   = df["house_price"].pct_change(4).mul(100)
df["income_to_price"] = df["avg_income"] / df["house_price"]
df["real_rate"]       = df["bank_rate"] - df["inflation"]

df_model = df.dropna().reset_index(drop=True)

FEATURES = [
    "time_idx", "time_idx_sq", "quarter",
    "bank_rate", "inflation", "unemployment",
    "avg_income", "income_to_price", "real_rate",
    "price_lag1", "price_lag2", "price_lag4",
    "price_roll4", "price_roll8", "price_yoy_pct",
]
TARGET = "house_price"

print(f"Records after lag trimming : {len(df_model)}")
print(f"Feature count              : {len(FEATURES)}")
print(f"Features used              : {FEATURES}")

# ============================================================
# SECTION 5 – TRAIN / TEST SPLIT  (chronological 80/20)
# ============================================================
print("\n" + "=" * 65)
print("SECTION 5: Train / Test Split")
print("=" * 65)

split_idx = int(len(df_model) * 0.80)
train     = df_model.iloc[:split_idx]
test      = df_model.iloc[split_idx:]
X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"Train : {len(train)} quarters  "
      f"({train['date'].min().date()} → {train['date'].max().date()})")
print(f"Test  : {len(test)} quarters   "
      f"({test['date'].min().date()} → {test['date'].max().date()})")

# ============================================================
# SECTION 6 – MODEL B: QUARTERLY PRICE FORECAST
#             Train all 4 models, compare accuracy
# ============================================================
print("\n" + "=" * 65)
print("SECTION 6: Model B — Quarterly Price Forecast (4 Algorithms)")
print("=" * 65)

MODELS = {
    "Linear Regression" : LinearRegression(),
    "Ridge Regression"  : Ridge(alpha=10.0),
    "Random Forest"     : RandomForestRegressor(n_estimators=300, max_depth=8,
                                                min_samples_leaf=2, random_state=42),
    "Gradient Boosting" : GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                                     learning_rate=0.05, subsample=0.8,
                                                     random_state=42),
}

def evaluate(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = r2_score(y_true, y_pred)
    print(f"  {label:<22}  MAE=£{mae:>7,.0f}  RMSE=£{rmse:>7,.0f}  "
          f"MAPE={mape:>5.1f}%  R²={r2:>6.3f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

results = {}
preds   = {}
for name, model in MODELS.items():
    model.fit(X_train, y_train)
    pred          = np.clip(model.predict(X_test), 0, None)
    results[name] = evaluate(y_test, pred, name)
    preds[name]   = pred

best_name  = min(results, key=lambda m: results[m]["RMSE"])
best_model = MODELS[best_name]
print(f"\n✓ Best model → {best_name}  "
      f"(RMSE=£{results[best_name]['RMSE']:,.0f}  "
      f"MAPE={results[best_name]['MAPE']:.1f}%  "
      f"R²={results[best_name]['R2']:.3f})")

# ============================================================
# SECTION 7 – TIME-SERIES CROSS-VALIDATION (5-fold)
# ============================================================
print("\n" + "=" * 65)
print("SECTION 7: Time-Series Cross-Validation (5-fold)")
print("=" * 65)

tscv    = TimeSeriesSplit(n_splits=5)
X_all   = df_model[FEATURES]
y_all   = df_model[TARGET]
cv_mod  = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=42)
cv_rmse, cv_mape, cv_r2 = [], [], []

for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_all), 1):
    cv_mod.fit(X_all.iloc[tr_idx], y_all.iloc[tr_idx])
    prd  = np.clip(cv_mod.predict(X_all.iloc[va_idx]), 0, None)
    rmse = np.sqrt(mean_squared_error(y_all.iloc[va_idx], prd))
    mape = np.mean(np.abs((y_all.iloc[va_idx] - prd) / y_all.iloc[va_idx])) * 100
    r2   = r2_score(y_all.iloc[va_idx], prd)
    cv_rmse.append(rmse); cv_mape.append(mape); cv_r2.append(r2)
    print(f"  Fold {fold}: RMSE=£{rmse:>7,.0f}   MAPE={mape:>5.1f}%   R²={r2:.3f}")

print(f"\n  Mean CV  RMSE = £{np.mean(cv_rmse):,.0f}  ±  £{np.std(cv_rmse):,.0f}")
print(f"  Mean CV  MAPE =  {np.mean(cv_mape):.1f}%  ±   {np.std(cv_mape):.1f}%")
print(f"  Mean CV  R²   =  {np.mean(cv_r2):.3f}")

# ============================================================
# SECTION 8 – FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 65)
print("SECTION 8: Feature Importance")
print("=" * 65)

best_model.fit(X_train, y_train)
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=FEATURES)
else:
    importances = pd.Series(np.abs(best_model.coef_), index=FEATURES)
importances = importances.sort_values(ascending=False)
print(f"\nFeature importances ({best_name}):")
print(importances.apply(lambda v: f"{v:.4f}").to_string())

# ============================================================
# SECTION 9 – FORECAST 2026–2030
# ============================================================
print("\n" + "=" * 65)
print("SECTION 9: Forecast — Littlehampton House Prices 2026–2030")
print("=" * 65)

# Retrain on ALL data before forecasting
best_model.fit(X_all, y_all)

FC_ASSUMPTIONS = {
    "bank_rate"   : [3.75,3.50,3.25,3.00, 2.75,2.75,3.00,3.00,
                     3.00,3.00,3.00,3.00, 3.00,3.00,3.00,3.00, 3.00,3.00,3.00,3.00],
    "inflation"   : [2.8,2.6,2.4,2.2, 2.1,2.0,2.0,2.0,
                     2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
    "unemployment": [4.3,4.3,4.2,4.2, 4.2,4.1,4.1,4.0,
                     4.0,4.0,4.0,4.0, 4.0,4.0,4.0,4.0, 4.0,4.0,4.0,4.0],
    "avg_income"  : [56000,56500,57000,57500, 58500,59500,60500,61500,
                     62500,63500,64500,65500, 66500,67500,68500,69500,
                     70500,71500,72500,73500],
}
FC_DATES = pd.date_range("2026-01-01", periods=20, freq="QS")
FC_YEARS = [2026]*4 + [2027]*4 + [2028]*4 + [2029]*4 + [2030]*4
FC_QTRS  = [1,2,3,4] * 5

history   = df_model.copy()
fc_rows   = []
last_tidx = df_model["time_idx"].iloc[-1]

for i in range(20):
    t    = last_tidx + i + 1
    yr   = FC_YEARS[i];  qt = FC_QTRS[i]
    br   = FC_ASSUMPTIONS["bank_rate"][i]
    inf  = FC_ASSUMPTIONS["inflation"][i]
    ue   = FC_ASSUMPTIONS["unemployment"][i]
    inc  = FC_ASSUMPTIONS["avg_income"][i]
    p1   = history["house_price"].iloc[-1]
    p2   = history["house_price"].iloc[-2]
    p4   = history["house_price"].iloc[-4]
    r4   = history["house_price"].iloc[-4:].mean()
    r8   = history["house_price"].iloc[-8:].mean()
    yoy  = ((p1 - p4) / p4) * 100 if p4 > 0 else 0
    i2p  = inc / p1 if p1 > 0 else 0
    rr   = br - inf

    feat = pd.DataFrame([{
        "time_idx": t, "time_idx_sq": t**2, "quarter": qt,
        "bank_rate": br, "inflation": inf, "unemployment": ue,
        "avg_income": inc, "income_to_price": i2p, "real_rate": rr,
        "price_lag1": p1, "price_lag2": p2, "price_lag4": p4,
        "price_roll4": r4, "price_roll8": r8, "price_yoy_pct": yoy,
    }])[FEATURES]

    pred_price = float(np.clip(best_model.predict(feat), 50_000, None))
    row = {"date": FC_DATES[i], "year": yr, "quarter": qt,
           "bank_rate": br, "inflation": inf, "unemployment": ue,
           "avg_income": inc, "house_price": pred_price, "time_idx": t}
    fc_rows.append(row)
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

forecast_df = pd.DataFrame(fc_rows)

print(f"\n{'Year':>5} {'Q':>2}  {'Predicted':>14}  {'YoY':>8}")
print("-" * 38)
for i, row in forecast_df.iterrows():
    yr  = int(row["year"]);  qt = int(row["quarter"])
    pr  = int(row["house_price"])
    # YoY vs same quarter previous year (real or forecast)
    prior_real = df_model[(df_model["year"]==yr-1) & (df_model["quarter"]==qt)]
    prior_fc   = forecast_df[(forecast_df["year"]==yr-1) & (forecast_df["quarter"]==qt)]
    if not prior_real.empty:
        yoy_s = f"{((pr-prior_real['house_price'].values[0])/prior_real['house_price'].values[0])*100:+.1f}%"
    elif not prior_fc.empty:
        yoy_s = f"{((pr-prior_fc['house_price'].values[0])/prior_fc['house_price'].values[0])*100:+.1f}%"
    else:
        yoy_s = "  n/a"
    print(f"  {yr}  Q{qt}  £{pr:>12,}  {yoy_s:>8}")

annual_avg = forecast_df.groupby("year")["house_price"].mean().apply(int)
print(f"\nAnnual average forecast:")
for yr, avg in annual_avg.items():
    base = df_model[df_model["year"]==yr-1]["house_price"].mean()
    if np.isnan(base):
        base = annual_avg.get(yr-1, avg)
    chg = ((avg - base) / base) * 100
    print(f"  {yr}  →  £{avg:,}   ({chg:+.1f}% vs {yr-1})")

# ============================================================
# SECTION 10 – CHARTS
# ============================================================
print("\n" + "=" * 65)
print("SECTION 10: Generating Charts")
print("=" * 65)

fig = plt.figure(figsize=(18, 26))
fig.suptitle(
    f"Littlehampton, West Sussex — Residential House Price Prediction\n"
    f"AI Model (Scikit-learn)  ·  Source: {data_source}\n"
    f"Historical 2000–2025  |  Forecast 2026–2030",
    fontsize=14, fontweight="bold", y=0.997
)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.38)


# ── Chart 1: Full history + test predictions + forecast ──────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.fill_between(df_model["date"], df_model["house_price"], alpha=0.1, color=BLUE)
ax1.plot(df_model["date"], df_model["house_price"],
         color=BLUE, linewidth=2, label="Historical (actual)", zorder=3)
ax1.plot(test["date"], preds[best_name],
         color=ORANGE, linewidth=2.2, linestyle="--",
         label=f"Test predictions — {best_name}", zorder=4)
ax1.plot(forecast_df["date"], forecast_df["house_price"],
         color=RED, linewidth=2.5, linestyle="--",
         marker="o", markersize=6, label="Forecast 2026–2030", zorder=5)
ax1.axvline(test["date"].iloc[0], color=GREY, linestyle=":",
            linewidth=1.5, label="Train/Test split")
ax1.axvline(forecast_df["date"].iloc[0], color=RED, linestyle=":",
            linewidth=1.5, alpha=0.5, label="Forecast start")
ax1.set_title("Littlehampton Average House Price — Historical & Forecast", fontsize=13)
ax1.set_ylabel("Average Price (£)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{int(x):,}"))
ax1.legend(fontsize=9)
ax1.grid(linestyle="--", alpha=0.35)
ax1.tick_params(axis="x", rotation=30)


# ── Chart 2: Model RMSE comparison ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
mnames   = list(results.keys())
rmse_v   = [results[m]["RMSE"] for m in mnames]
bar_cols = [GREEN if m == best_name else BLUE for m in mnames]
bars2 = ax2.barh(mnames, rmse_v, color=bar_cols, alpha=0.85)
for bar, val in zip(bars2, rmse_v):
    ax2.text(val + 200, bar.get_y() + bar.get_height()/2,
             f"£{val:,.0f}", va="center", fontsize=9)
ax2.set_xlabel("RMSE (£)")
ax2.set_title("Model Comparison — RMSE\n(Green = best model)", fontsize=11)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{int(x):,}"))
ax2.grid(axis="x", linestyle="--", alpha=0.35)


# ── Chart 3: R² & MAPE comparison ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
x_pos  = np.arange(len(mnames))
r2_v   = [results[m]["R2"]   for m in mnames]
mape_v = [results[m]["MAPE"] for m in mnames]
ax3b   = ax3.twinx()
bars_r = ax3.bar(x_pos - 0.2, r2_v,  0.35, color=BLUE,   alpha=0.80, label="R² (left)")
bars_m = ax3b.bar(x_pos + 0.2, mape_v, 0.35, color=ORANGE, alpha=0.80, label="MAPE % (right)")
for bar, val in zip(bars_r, r2_v):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.005,
             f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
for bar, val in zip(bars_m, mape_v):
    ax3b.text(bar.get_x() + bar.get_width()/2, val + 0.1,
              f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold", color=ORANGE)
ax3.set_xticks(x_pos)
ax3.set_xticklabels([m.replace(" ","\n") for m in mnames], fontsize=9)
ax3.set_ylim(0, 1.15)
ax3.set_ylabel("R² Score", color=BLUE)
ax3b.set_ylabel("MAPE (%)", color=ORANGE)
ax3.set_title("R²  vs  MAPE — All Models", fontsize=11)
ax3.grid(axis="y", linestyle="--", alpha=0.35)
lines  = [bars_r[0], bars_m[0]]
labels = ["R² (left axis)", "MAPE % (right axis)"]
ax3.legend(lines, labels, fontsize=8, loc="upper right")


# ── Chart 4: Residuals — best model ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
residuals = y_test.values - preds[best_name]
ax4.scatter(preds[best_name], residuals,
            alpha=0.7, color=PURPLE, edgecolors="white", linewidths=0.5, s=70)
ax4.axhline(0, color=RED, linewidth=1.8, linestyle="--")
z     = np.polyfit(preds[best_name], residuals, 1)
xline = np.linspace(preds[best_name].min(), preds[best_name].max(), 200)
ax4.plot(xline, np.poly1d(z)(xline), color=ORANGE, linewidth=1.5,
         linestyle=":", label="Trend line")
ax4.set_xlabel("Predicted Price (£)")
ax4.set_ylabel("Residual  Actual − Predicted (£)")
ax4.set_title(f"Residual Plot — {best_name}", fontsize=11)
ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{int(x):,}"))
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{int(x):,}"))
ax4.legend(fontsize=8)
ax4.grid(linestyle="--", alpha=0.35)


# ── Chart 5: Feature importance ───────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
top12    = importances.head(12)
fi_cols  = [GREEN if i == 0 else BLUE for i in range(len(top12))]
ax5.barh(top12.index[::-1], top12.values[::-1], color=fi_cols[::-1], alpha=0.85)
ax5.set_xlabel("Importance Score")
ax5.set_title(f"Top 12 Feature Importances\n({best_name})", fontsize=11)
ax5.grid(axis="x", linestyle="--", alpha=0.35)


# ── Chart 6: Annual average bar — historical + forecast ───────────────────────
ax6 = fig.add_subplot(gs[3, :])
hist_ann = df_model.groupby("year")["house_price"].mean().reset_index()
hist_ann.columns = ["year", "avg_price"]
fc_ann   = annual_avg.reset_index()
fc_ann.columns = ["year", "avg_price"]
all_ann  = pd.concat([hist_ann, fc_ann], ignore_index=True)
bar_c    = [ORANGE if y >= 2026 else BLUE for y in all_ann["year"]]
ax6.bar(all_ann["year"], all_ann["avg_price"], color=bar_c, alpha=0.85, width=0.75)
for _, row in fc_ann.iterrows():
    ax6.text(row["year"], row["avg_price"] + 2000,
             f"£{int(row['avg_price']):,}", ha="center", fontsize=8.5,
             fontweight="bold", color=ORANGE)
ax6.axvline(2025.5, color=RED, linestyle="--", linewidth=1.5, label="Forecast boundary")
ax6.set_xlabel("Year")
ax6.set_ylabel("Annual Average House Price (£)")
ax6.set_title(
    "Littlehampton Annual Average House Price  "
    "(Blue = Historical  |  Orange = Forecast 2026–2030)",
    fontsize=12
)
ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{int(x):,}"))
ax6.legend(fontsize=10)
ax6.grid(axis="y", linestyle="--", alpha=0.35)
ax6.tick_params(axis="x", rotation=45)


chart_file = "littlehampton_housing_prediction.png"
plt.savefig(chart_file, dpi=150, bbox_inches="tight")
plt.close()
print(f"[SAVED] Chart → {os.path.abspath(chart_file)}")
print(f"\nModel complete.  Data source: {data_source}")
