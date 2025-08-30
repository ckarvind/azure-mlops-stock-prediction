# Phase 8 — Backtesting & Metrics (production)
import os, io, argparse, json
import numpy as np
import pandas as pd

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

# =========================
# Args
# =========================
p = argparse.ArgumentParser()
# infra
p.add_argument("--vault_url", required=True)
p.add_argument("--secret_name", default="AzureBlobConnStr")
p.add_argument("--container", required=True)

# locations
p.add_argument("--model_ready_folder", default="model_ready_data")       # per-ticker price files live here
p.add_argument("--predictions_folder", default="predictions")            # where final_scored_* lives
p.add_argument("--final_scored_prefix", default="predictions/final_scored")
p.add_argument("--output_folder", default="backtest_outputs")

# knobs
p.add_argument("--initial_capital", type=float, default=100000.0)
p.add_argument("--top_n", type=int, default=10)
p.add_argument("--rebalance_freq", choices=["Q", "M", "W"], default="Q")  # Quarterly / Monthly / Weekly
p.add_argument("--splits", default="val,test")                            # which final_scored_{split}.parquet to use
p.add_argument("--rf_rate", type=float, default=0.02)                     # annual risk-free (for Sharpe/Sortino)

args = p.parse_args()

# =========================
# KV + Blob
# =========================
print("🔑 Using Key Vault -> Blob connection string")
cred = DefaultAzureCredential()
conn = SecretClient(vault_url=args.vault_url, credential=cred).get_secret(args.secret_name).value
bs = BlobServiceClient.from_connection_string(conn)
cc = bs.get_container_client(args.container)

def blob_exists(path: str) -> bool:
    try:
        cc.get_blob_client(path).get_blob_properties()
        return True
    except Exception:
        return False

def read_parquet_blob(path: str) -> pd.DataFrame:
    print(f"📥 {path}")
    b = cc.download_blob(path).readall()
    return pd.read_parquet(io.BytesIO(b))

def upload_bytes(data: bytes, path: str):
    cc.upload_blob(path, data, overwrite=True)
    print(f"☁️ Uploaded: {path}")

def upload_df_csv(df: pd.DataFrame, path: str):
    b = df.to_csv(index=False).encode("utf-8")
    upload_bytes(b, path)

# =========================
# Metrics
# =========================
def calculate_cagr(df: pd.DataFrame) -> float:
    start_val = df["portfolio_value"].iloc[0]
    end_val   = df["portfolio_value"].iloc[-1]
    years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    return (end_val / start_val) ** (1 / years) - 1 if years > 0 else np.nan

def calculate_sharpe(df: pd.DataFrame, rf_rate: float = 0.02) -> float:
    # daily comp: assume 252 trading days
    rets = df["portfolio_value"].pct_change().dropna()
    if rets.std() == 0 or rets.empty:
        return np.nan
    excess = rets - rf_rate / 252.0
    return np.sqrt(252.0) * excess.mean() / excess.std()

def calculate_sortino(df: pd.DataFrame, rf_rate: float = 0.02) -> float:
    rets = df["portfolio_value"].pct_change().dropna()
    if rets.empty:
        return np.nan
    downside = rets[rets < 0]
    if downside.std() == 0:
        return np.nan
    excess = rets - rf_rate / 252.0
    return np.sqrt(252.0) * excess.mean() / downside.std()

def calculate_mdd(df: pd.DataFrame) -> float:
    roll_max = df["portfolio_value"].cummax()
    drawdown = (df["portfolio_value"] - roll_max) / roll_max
    return float(drawdown.min()) if not drawdown.empty else np.nan

# =========================
# Load ranking (final_scored) for requested splits
# =========================
splits = [s.strip() for s in args.splits.split(",") if s.strip()]
frames = []
for sp in splits:
    path = f"{args.final_scored_prefix}_{sp}.parquet"
    if not blob_exists(path):
        print(f"⚠️ Missing final scored for split='{sp}' at {path}; skipping this split.")
        continue
    df = read_parquet_blob(path)
    frames.append(df)

if not frames:
    raise FileNotFoundError("No final_scored parquet found for requested splits.")

ranking_df = pd.concat(frames, ignore_index=True)
rename_map = {
    "Date": "date",
    "Ticker": "ticker",
    "model1_prob_m1": "model1_prob",
    "model2_pred_return_m2": "model2_pred_return",
}
ranking_df = ranking_df.rename(columns={k: v for k, v in rename_map.items() if k in ranking_df.columns})
ranking_df["date"] = pd.to_datetime(ranking_df["date"])
ranking_df = ranking_df.sort_values(["date", "final_score"], ascending=[True, False]).reset_index(drop=True)

# =========================
# Map tickers -> price parquet path (from model_ready_folder)
# =========================
print("📂 Building ticker → file map from model_ready_folder ...")
ticker_files = {}
for b in cc.list_blobs(name_starts_with=f"{args.model_ready_folder}/"):
    if not b.name.endswith(".parquet"):
        continue
    base = os.path.basename(b.name)
    # assumes file name starts with TICKER_*.parquet
    ticker = base.split("_")[0]
    ticker_files.setdefault(ticker, b.name)

print(f"✅ Found price files for ~{len(ticker_files)} tickers")

# cache for price & sector
price_cache = {}
sector_cache = {}

def get_price_df(ticker: str):
    if ticker in price_cache:
        return price_cache[ticker]
    path = ticker_files.get(ticker)
    if not path:
        return None
    df = read_parquet_blob(path)
    # normalize columns
    df = df.rename(columns={"Date":"date", "Adj Close":"Adj Close", "adj_close":"adj_close", "Close":"Close", "close":"close"})
    df["date"] = pd.to_datetime(df["date"])
    price_cache[ticker] = df.sort_values("date").reset_index(drop=True)
    # sector (best-effort)
    sector = None
    for cand in ["Sector","sector"]:
        if cand in df.columns and pd.notnull(df[cand]).any():
            sector = df[cand].dropna().iloc[0]
            break
    sector_cache[ticker] = sector if sector else "Unknown"
    return price_cache[ticker]

def pick_price_col(df: pd.DataFrame):
    for c in ["Adj Close","adj_close","Close","close"]:
        if c in df.columns:
            return c
    raise KeyError("No price column among ['Adj Close','adj_close','Close','close']")

# =========================
# Rebalancing calendar
# =========================
available_dates = sorted(ranking_df["date"].unique())
if not available_dates:
    raise ValueError("No dates in ranking_df")

if args.rebalance_freq == "Q":
    rebalance_dates = pd.date_range(start=available_dates[0], end=available_dates[-1], freq="QS")
elif args.rebalance_freq == "M":
    rebalance_dates = pd.date_range(start=available_dates[0], end=available_dates[-1], freq="MS")
elif args.rebalance_freq == "W":
    rebalance_dates = pd.date_range(start=available_dates[0], end=available_dates[-1], freq="W")
else:
    rebalance_dates = pd.to_datetime(available_dates)

# =========================
# Backtest loop
# =========================
portfolio_value = float(args.initial_capital)
portfolio_history = []
trade_signals = []
current_holdings = []

for i, rebalance_date in enumerate(rebalance_dates):
    # nearest available ranking date >= scheduled rebalance_date
    day_df = ranking_df[ranking_df["date"] >= rebalance_date]
    if day_df.empty:
        continue
    rebalance_actual_date = day_df["date"].min()

    # Top-N at that date
    top_stocks = (
        day_df[day_df["date"] == rebalance_actual_date]
        .sort_values("final_score", ascending=False)
        .head(args.top_n)
    )
    new_holdings = [str(t) for t in top_stocks["ticker"].tolist()]

    # Determine trades
    buys  = [t for t in new_holdings if t not in current_holdings]
    sells = [t for t in current_holdings if t not in new_holdings]
    holds = [t for t in new_holdings if t in current_holdings]

    # Ensure price/sector cached for used tickers
    for ticker in set(new_holdings + sells):
        get_price_df(ticker)  # fills caches if exists

    # Record trade signals
    for ticker, action in [(t,"BUY") for t in buys] + [(t,"HOLD") for t in holds] + [(t,"SELL") for t in sells]:
        trade_signals.append({
            "rebalance_date": rebalance_actual_date,
            "ticker": ticker,
            "sector": sector_cache.get(ticker, "Unknown"),
            "action": action
        })

    # Apply holdings
    current_holdings = new_holdings
    equal_allocation = portfolio_value / len(current_holdings) if current_holdings else 0.0

    # Advance until next rebalance date (exclusive) or end
    next_rebalance_date = (
        rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else available_dates[-1]
    )

    current_date = pd.Timestamp(rebalance_actual_date)
    while current_date < next_rebalance_date:
        day_returns = []

        # compute returns only on **trading days** (avoid repeating same step on non-trading days)
        for ticker in current_holdings:
            dfp = price_cache.get(ticker)
            if dfp is None:
                continue
            price_col = pick_price_col(dfp)
            # index where dfp['date'] == current_date
            # if the date is not a trading day for this ticker, skip it
            m = dfp["date"] == current_date
            if not m.any():
                continue
            idx = int(np.flatnonzero(m)[0])
            if idx + 1 >= len(dfp):
                continue
            today_price = float(dfp.iloc[idx][price_col])
            next_price  = float(dfp.iloc[idx + 1][price_col])
            if today_price > 0:
                day_returns.append((next_price - today_price) / today_price)

        # update portfolio if at least one ticker has a trading day today
        if day_returns:
            portfolio_value *= (1.0 + float(np.mean(day_returns)))

        portfolio_history.append({"date": current_date, "portfolio_value": portfolio_value})
        current_date += pd.Timedelta(days=1)

# =========================
# Results & uploads
# =========================
results_df = pd.DataFrame(portfolio_history).drop_duplicates(subset="date").sort_values("date")
if results_df.empty:
    raise ValueError("Portfolio history is empty — check date alignment and price files.")

cagr   = calculate_cagr(results_df)
sharpe = calculate_sharpe(results_df, rf_rate=args.rf_rate)
sortino= calculate_sortino(results_df, rf_rate=args.rf_rate)
mdd    = calculate_mdd(results_df)

summary_df = pd.DataFrame({
    "CAGR": [cagr],
    "Sharpe": [sharpe],
    "Sortino": [sortino],
    "Max_Drawdown": [mdd],
    "Initial_Capital": [args.initial_capital],
    "Top_N": [args.top_n],
    "Rebalance_Freq": [args.rebalance_freq],
    "Splits": [",".join(splits)]
})

# Save locally (for MLflow) and upload to Blob
results_csv = "backtest_results.csv"
signals_csv = "trade_signals.csv"
summary_csv = "backtest_summary.csv"

results_df.to_csv(results_csv, index=False)
pd.DataFrame(trade_signals).to_csv(signals_csv, index=False)
summary_df.to_csv(summary_csv, index=False)

upload_df_csv(results_df, f"{args.output_folder}/backtest_results.csv")
upload_df_csv(pd.DataFrame(trade_signals), f"{args.output_folder}/trade_signals.csv")
upload_df_csv(summary_df, f"{args.output_folder}/backtest_summary.csv")

# =========================
# MLflow logging
# =========================
try:
    import mlflow
    if os.getenv("MLFLOW_TRACKING_URI", "").startswith("azureml://") or "AZUREML_RUN_ID" in os.environ:
        backend = "azureml"
    else:
        mlflow.set_tracking_uri(f"file:{os.path.join(os.getcwd(),'mlruns')}")
        backend = "file"
    print(f"🧪 MLflow backend = {backend}")

    with mlflow.start_run():
        mlflow.log_params({
            "initial_capital": args.initial_capital,
            "top_n": args.top_n,
            "rebalance_freq": args.rebalance_freq,
            "splits": ",".join(splits),
            "rf_rate": args.rf_rate
        })
        mlflow.log_metrics({
            "CAGR": float(cagr) if cagr is not None else np.nan,
            "Sharpe": float(sharpe) if sharpe is not None else np.nan,
            "Sortino": float(sortino) if sortino is not None else np.nan,
            "Max_Drawdown": float(mdd) if mdd is not None else np.nan
        })
        # artifacts
        mlflow.log_artifact(results_csv)
        mlflow.log_artifact(signals_csv)
        mlflow.log_artifact(summary_csv)

        # also persist summary JSON artifact
        with open("backtest_summary.json", "w") as f:
            json.dump(summary_df.to_dict(orient="list"), f, indent=2, default=lambda x: None)
        mlflow.log_artifact("backtest_summary.json")
except Exception as e:
    print(f"⚠️ MLflow logging skipped: {e}")

print("✅ Backtest complete & uploaded to Blob.")
