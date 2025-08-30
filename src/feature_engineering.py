#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 — Feature Engineering + Post-Processing with Sector Mapping (production)

- Reads per-ticker OHLCV parquet files from Blob (e.g., stockdata_us_adjclose/)
- Adds TA indicators, returns/volatility, sector, dividend/split flags
- Adds robust z-score outlier flags for all numeric cols
- Writes model-ready parquet per ticker to Blob (e.g., model_ready_data/TICKER_YYYYMMDD.parquet)
- Auth: Key Vault -> (secret) -> Blob connection string
"""

import os, io, sys, time, argparse, logging
from io import BytesIO
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
from scipy.stats import zscore

# ---- 3rd party (must be in env) ----
# NOTE: ensure 'pandas_ta' is installed in your environment (requirements.txt / conda env)
import pandas_ta as ta

# ---- Azure SDKs ----
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceModifiedError, HttpResponseError

# =========================
# CLI
# =========================
p = argparse.ArgumentParser(description="Phase 2 — Feature Engineering (production)")
# infra
p.add_argument("--vault_url", required=True, help="Key Vault URL, e.g., https://<vault>.vault.azure.net")
p.add_argument("--secret_name", default="AzureBlobConnStr", help="KV secret name storing Blob conn str")
p.add_argument("--container", required=True, help="Blob container, e.g., stock-data")

# I/O
p.add_argument("--input_prefix", default="stockdata_us_adjclose", help="Blob folder with raw per-ticker parquet")
p.add_argument("--output_prefix", default="model_ready_data", help="Blob folder for model-ready outputs")
p.add_argument("--sector_blob", default="sectors_20250809.csv", help="CSV blob with Ticker,Sector mapping")

# knobs
p.add_argument("--z_threshold", type=float, default=4.0, help="z-score threshold for outlier flags")
p.add_argument("--workers", type=int, default=min(cpu_count(), 12), help="parallel workers")
p.add_argument("--limit", type=int, default=0, help="process only first N input files (0 = all)")
p.add_argument("--version_ts", default=datetime.now().strftime("%Y%m%d"), help="version suffix in output names")
p.add_argument("--log_level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
p.add_argument("--mlflow", type=lambda s: s.lower()=="true", default=True, help="Log minimal metadata to MLflow")
args = p.parse_args()

# =========================
# Logging
# =========================
logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("phase2")

# =========================
# KV + Blob
# =========================
log.info("🔑 Fetching Blob connection string from Key Vault")
cred = DefaultAzureCredential()
conn = SecretClient(vault_url=args.vault_url, credential=cred).get_secret(args.secret_name).value
bs = BlobServiceClient.from_connection_string(conn)
cc = bs.get_container_client(args.container)

def list_parquet(prefix: str) -> List[str]:
    prefix = prefix.strip("/")+ "/"
    out = []
    for b in cc.list_blobs(name_starts_with=prefix):
        if b.name.lower().endswith(".parquet"):
            out.append(b.name)
    return out

def download_blob(path: str, attempts: int = 6) -> bytes:
    for i in range(attempts):
        try:
            return cc.download_blob(path, max_concurrency=1).readall()
        except (ResourceModifiedError, HttpResponseError) as e:
            wait = min(2 ** i, 30)
            log.warning(f"⚠️ Read retry {i+1}/{attempts} for {path} in {wait}s ({e})")
            time.sleep(wait)
    raise RuntimeError(f"Failed to read {path} after {attempts} attempts")

def upload_bytes(data: bytes, path: str):
    cc.upload_blob(name=path, data=data, overwrite=True)
    log.info(f"☁️ Uploaded: {path}")

def upload_parquet(df: pd.DataFrame, path: str):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    upload_bytes(buf.getvalue(), path)

# =========================
# Load Sector Map
# =========================
log.info(f"📥 Loading sector mapping: {args.sector_blob}")
sector_csv = download_blob(args.sector_blob)
sector_df = pd.read_csv(io.BytesIO(sector_csv))
# normalize columns
sector_cols = {c.lower(): c for c in sector_df.columns}
ticker_col = sector_cols.get("ticker", "Ticker")
sector_col = sector_cols.get("sector", "Sector")
sector_df = sector_df.rename(columns={ticker_col: "Ticker", sector_col: "Sector"})
sector_map: Dict[str, str] = dict(zip(sector_df["Ticker"].astype(str), sector_df["Sector"].astype(str)))
log.info(f"✅ Sector mapping entries: {len(sector_map):,}")

# =========================
# Core per-file transform
# =========================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Expecting OHLCV columns; be tolerant about case
    ren = {c: c.strip() for c in df.columns}
    df = df.rename(columns=ren)

    # Ensure standard keys
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("Input file missing required columns: 'Date', 'Ticker'")

    # Sorting
    df = df.sort_values("Date").reset_index(drop=True)

    # Price column preference
    price_col = next((c for c in ["Adj Close","adj_close","Close","close"] if c in df.columns), None)
    if price_col is None:
        raise KeyError("No price column among ['Adj Close','adj_close','Close','close']")

    # TA indicators
    # Simple moving averages & momentum
    df["SMA_20"] = ta.sma(df[price_col], length=20)
    df["EMA_20"] = ta.ema(df[price_col], length=20)
    df["RSI_14"] = ta.rsi(df[price_col], length=14)

    macd = ta.macd(df[price_col], fast=12, slow=26)
    if macd is not None and isinstance(macd, pd.DataFrame):
        # typical columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        for k in macd.columns:
            df[k] = macd[k]
        df["MACD"] = df.get("MACD_12_26_9", df.get("MACD", np.nan))
        df["MACD_signal"] = df.get("MACDs_12_26_9", df.get("MACD_signal", np.nan))

    bb = ta.bbands(df[price_col], length=20)
    if bb is not None and isinstance(bb, pd.DataFrame):
        # typical columns: BBU_20_2.0, BBM_20_2.0, BBL_20_2.0
        for k in bb.columns:
            df[k] = bb[k]
        df["BB_upper"] = df.get("BBU_20_2.0", df.get("BBU_20_2", np.nan))
        df["BB_lower"] = df.get("BBL_20_2.0", df.get("BBL_20_2", np.nan))

    # ATR/ADX (need High/Low/Close)
    hi = next((c for c in ["High","high"] if c in df.columns), None)
    lo = next((c for c in ["Low","low"] if c in df.columns), None)
    cl = next((c for c in ["Close","close", price_col] if c in df.columns), None)
    if hi and lo and cl:
        df["ATR_14"] = ta.atr(df[hi], df[lo], df[cl], length=14)
        adx = ta.adx(df[hi], df[lo], df[cl], length=14)
        if adx is not None and isinstance(adx, pd.DataFrame):
            df["ADX_14"] = adx.get("ADX_14", np.nan)

    # Returns & vol
    df["daily_return"] = df[price_col].pct_change()
    df["volatility_20d"] = df["daily_return"].rolling(window=20, min_periods=20).std()

    # drop initial NaNs created by indicators
    df = df.dropna().reset_index(drop=True)
    return df

def add_events_sector_and_outliers(df: pd.DataFrame, z_thr: float, sector_map: Dict[str,str]) -> pd.DataFrame:
    # Sector
    tkr = str(df["Ticker"].iloc[0])
    df["Sector"] = sector_map.get(tkr, "Unknown")

    # Dividend / Split events
    if "Dividends" in df.columns:
        df["Dividend_event"] = (df["Dividends"] > 0).astype(np.int8)
    else:
        df["Dividend_event"] = 0
    if "Stock Splits" in df.columns:
        df["StockSplit_event"] = (df["Stock Splits"] > 0).astype(np.int8)
    else:
        df["StockSplit_event"] = 0

    # Outlier flags on all numeric cols (robust to NaNs/constant)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if not c.endswith("_is_outlier")]
    for col in num_cols:
        arr = df[col].astype(float).to_numpy(copy=False)
        if not np.isfinite(arr).any() or np.nanstd(arr) == 0:
            df[f"{col}_is_outlier"] = 0
            continue
        z = zscore(arr, nan_policy="omit")
        mask = np.zeros(len(arr), dtype=np.int8)
        valid = ~np.isnan(z)
        mask[valid] = (np.abs(z[valid]) > z_thr).astype(np.int8)
        df[f"{col}_is_outlier"] = mask

    # Drop large raw columns we do not model directly
    drop_cols = [c for c in ["Dividends","Stock Splits"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df

def process_one(blob_name: str) -> Tuple[str, bool, Optional[str]]:
    try:
        # read parquet
        raw = download_blob(blob_name)
        df = pd.read_parquet(BytesIO(raw))
        df = compute_indicators(df)
        df = add_events_sector_and_outliers(df, args.z_threshold, sector_map)

        tkr = str(df["Ticker"].iloc[0])
        out_name = f"{args.output_prefix.strip('/')}/{tkr}_{args.version_ts}.parquet"
        upload_parquet(df, out_name)
        return tkr, True, None
    except Exception as e:
        return blob_name, False, f"{type(e).__name__}: {e}"

# =========================
# Discover & Run
# =========================
inputs = list_parquet(args.input_prefix)
if args.limit and args.limit > 0:
    inputs = inputs[:args.limit]
log.info(f"📂 Found {len(inputs)} parquet files under {args.input_prefix}/")

t0 = time.time()
ok, bad = [], []

if args.workers and args.workers > 1:
    with Pool(processes=args.workers) as pool:
        for tkr, success, err in pool.imap_unordered(process_one, inputs):
            (ok if success else bad).append((tkr, err))
            if success:
                log.info(f"✅ {tkr}")
            else:
                log.error(f"❌ {tkr} -> {err}")
else:
    for b in inputs:
        tkr, success, err = process_one(b)
        (ok if success else bad).append((tkr, err))
        if success:
            log.info(f"✅ {tkr}")
        else:
            log.error(f"❌ {tkr} -> {err}")

log.info(f"⏱ Done in {time.time()-t0:.1f}s | Success: {len(ok)} | Failed: {len(bad)}")

# =========================
# MLflow (minimal, Azure-safe)
# =========================
if args.mlflow:
    try:
        import mlflow
        if os.getenv("MLFLOW_TRACKING_URI","").startswith("azureml://") or "AZUREML_RUN_ID" in os.environ:
            backend = "azureml"
        else:
            mlflow.set_tracking_uri(f"file:{os.path.join(os.getcwd(),'mlruns')}")
            backend = "file"
        log.info(f"🧪 MLflow backend = {backend}")

        with mlflow.start_run():
            mlflow.log_params({
                "input_prefix": args.input_prefix,
                "output_prefix": args.output_prefix,
                "sector_blob": args.sector_blob,
                "z_threshold": args.z_threshold,
                "workers": args.workers,
                "version_ts": args.version_ts,
            })
            mlflow.log_metrics({
                "files_total": len(inputs),
                "files_success": len(ok),
                "files_failed": len(bad),
            })
    except Exception as e:
        log.warning(f"⚠️ MLflow logging skipped: {e}")

# =========================
# Exit code reflects failures (useful in CI/AML)
# =========================
sys.exit(0 if len(bad) == 0 else 2)
