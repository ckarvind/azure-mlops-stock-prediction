# Phase 4 — Training Data Preparation (production)
import os, io, argparse, json
import numpy as np
import pandas as pd

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

# ---------------- args ----------------
p = argparse.ArgumentParser()
# infra
p.add_argument("--vault_url", required=True)
p.add_argument("--secret_name", default="AzureBlobConnStr")
p.add_argument("--container", required=True)

# I/O
p.add_argument("--input_folder", default="model_ready_data")   # where per-ticker feature parquet files live
p.add_argument("--output_local", default="training_data")      # local temp output folder
p.add_argument("--output_blob",  default="training_data")      # blob prefix to upload splits
p.add_argument("--upload_to_blob", type=lambda s: s.lower()=="true", default=True)

# filtering / labels
p.add_argument("--tickers", default="ALL")                     # "ALL" or "AAPL,NVDA,MSFT"
p.add_argument("--start_date", default="1900-01-01")
p.add_argument("--end_date",   default="2100-12-31")
p.add_argument("--target_horizon_days", type=int, default=252) # ~12 months
p.add_argument("--high_growth_threshold", type=float, default=0.25)
p.add_argument("--drop_outlier_flags", type=lambda s: s.lower()=="true", default=False)

args = p.parse_args()
os.makedirs(args.output_local, exist_ok=True)

# ---------------- KV + Blob ----------------
print("🔑 Using Key Vault -> Blob connection string")
cred = DefaultAzureCredential()
conn = SecretClient(vault_url=args.vault_url, credential=cred).get_secret(args.secret_name).value
bs = BlobServiceClient.from_connection_string(conn)
cc = bs.get_container_client(args.container)

def list_parquet_under(prefix: str):
    for b in cc.list_blobs(name_starts_with=f"{prefix.rstrip('/')}/"):
        if b.name.lower().endswith(".parquet"):
            yield b.name

def read_parquet_blob(path: str) -> pd.DataFrame:
    print(f"📥 {path}")
    b = cc.download_blob(path).readall()
    return pd.read_parquet(io.BytesIO(b))

def upload_parquet(df: pd.DataFrame, path: str):
    buf = io.BytesIO(); df.to_parquet(buf, index=False); buf.seek(0)
    cc.upload_blob(name=path, data=buf.getvalue(), overwrite=True)
    print(f"☁️ Uploaded: {path}")

# ---------------- load & assemble ----------------
blob_names = list(list_parquet_under(args.input_folder))
if not blob_names:
    raise FileNotFoundError(f"No parquet files found under {args.input_folder}/ in container {args.container}")

dfs = [read_parquet_blob(n) for n in blob_names]
data = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded {len(data):,} rows from {len(blob_names)} files.")

# normalize keys & types
if "Date" not in data.columns or "Ticker" not in data.columns:
    raise ValueError("Input data must contain 'Date' and 'Ticker' columns.")
data["Date"] = pd.to_datetime(data["Date"])
data["Ticker"] = data["Ticker"].astype(str)
if "Sector" in data.columns:
    data["Sector"] = data["Sector"].astype(str)

# optional filters
tickers = None if args.tickers.upper() == "ALL" else [t.strip() for t in args.tickers.split(",") if t.strip()]
if tickers:
    before = len(data); data = data[data["Ticker"].isin(tickers)]; after = len(data)
    print(f"🎯 Filter tickers: kept {after:,}/{before:,}")
data = data[(data["Date"] >= args.start_date) & (data["Date"] <= args.end_date)].reset_index(drop=True)
print(f"🗓️ Date window [{args.start_date} .. {args.end_date}] -> {len(data):,} rows")

# enforce per-ticker chronological order
data = data.sort_values(["Ticker","Date"]).reset_index(drop=True)

# ---------------- label construction ----------------
# autodetect a price column
price_candidates = [c for c in ["Adj Close","adj_close","Close","close"] if c in data.columns]
if not price_candidates:
    raise KeyError("No price column found among ['Adj Close','adj_close','Close','close']")
price_col = price_candidates[0]
print(f"💲 Using price column: {price_col}")

# future_return = price[t+H]/price[t] - 1 (per ticker)
data["future_return"] = (
    data.groupby("Ticker")[price_col].shift(-args.target_horizon_days) / data[price_col] - 1
)
data["high_growth_label"] = (data["future_return"] >= args.high_growth_threshold).astype("Int64")

before = len(data)
data = data.dropna(subset=["future_return"]).reset_index(drop=True)
print(f"🧼 Dropped {before - len(data):,} tail rows with no future label (horizon={args.target_horizon_days}).")

# ---------------- optional: drop outlier flags ----------------
if args.drop_outlier_flags:
    outlier_cols = [c for c in data.columns if c.endswith("_is_outlier")]
    data = data.drop(columns=outlier_cols)
    print(f"🧹 Dropped {len(outlier_cols)} outlier-flag columns.")

# ---------------- one-hot encode Sector (if present) ----------------
if "Sector" in data.columns:
    try:
        from sklearn.preprocessing import OneHotEncoder
        try:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        mat = ohe.fit_transform(data[["Sector"]])
        sector_cols = [f"Sector_{cat}" for cat in ohe.categories_[0]]
        sector_df = pd.DataFrame(mat, columns=sector_cols, index=data.index)
        data = pd.concat([data.drop(columns=["Sector"]), sector_df], axis=1)
        print(f"🏷️ One-hot encoded Sector -> {len(sector_cols)} columns")
    except Exception as e:
        print(f"⚠️ Sector OHE skipped: {e}")

# ---------------- feature list & downcast ----------------
exclude = {"Date","Ticker","future_return","high_growth_label"}
feature_cols = [c for c in data.columns if c not in exclude]
print(f"🧩 Using {len(feature_cols)} features.")

# downcast to save space
for c in data.select_dtypes(include=["float64"]).columns:
    data[c] = data[c].astype(np.float32)
for c in data.select_dtypes(include=["int64"]).columns:
    if c != "high_growth_label":
        data[c] = data[c].astype(np.int32)

# save feature list locally and (optionally) to Blob
feat_csv_local = os.path.join(args.output_local, "feature_columns.csv")
pd.Series(feature_cols, name="features").to_csv(feat_csv_local, index=False)
if args.upload_to_blob:
    cc.upload_blob(name=f"{args.output_blob}/feature_columns.csv",
                   data=open(feat_csv_local,"rb").read(), overwrite=True)
    print("☁️ Uploaded: training_data/feature_columns.csv")

# ---------------- time split (70/15/15) per ticker ----------------
def split_by_time(df_t):
    n = len(df_t)
    i1 = int(0.70 * n)
    i2 = int(0.85 * n)
    return df_t.iloc[:i1], df_t.iloc[i1:i2], df_t.iloc[i2:]

train_parts, val_parts, test_parts = [], [], []
for tkr, g in data.groupby("Ticker", sort=False):
    g = g.sort_values("Date")
    tr, va, te = split_by_time(g)
    if len(tr):  train_parts.append(tr)
    if len(va):  val_parts.append(va)
    if len(te):  test_parts.append(te)

train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=data.columns)
val_df   = pd.concat(val_parts,   ignore_index=True) if val_parts   else pd.DataFrame(columns=data.columns)
test_df  = pd.concat(test_parts,  ignore_index=True) if test_parts  else pd.DataFrame(columns=data.columns)

print(f"📊 Splits -> Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

# ---------------- basic schema checks ----------------
required = {"Date","Ticker","future_return","high_growth_label"}
for name, df in [("train",train_df),("val",val_df),("test",test_df)]:
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"{name} split missing columns: {miss}")

# ---------------- save locally ----------------
train_path = os.path.join(args.output_local, "train.parquet")
val_path   = os.path.join(args.output_local, "val.parquet")
test_path  = os.path.join(args.output_local, "test.parquet")

train_df.to_parquet(train_path, index=False)
val_df.to_parquet(val_path, index=False)
test_df.to_parquet(test_path, index=False)

print(f"💾 Saved:\n - {train_path}\n - {val_path}\n - {test_path}")

# ---------------- upload to blob (optional) ----------------
if args.upload_to_blob:
    for local_fp, name in [(train_path,"train.parquet"),(val_path,"val.parquet"),(test_path,"test.parquet")]:
        with open(local_fp,"rb") as f:
            cc.upload_blob(name=f"{args.output_blob}/{name}", data=f.read(), overwrite=True)
            print(f"☁️ Uploaded: {args.output_blob}/{name}")

# ---------------- MLflow logging ----------------
try:
    import mlflow
    if os.getenv("MLFLOW_TRACKING_URI", "").startswith("azureml://") or "AZUREML_RUN_ID" in os.environ:
        backend = "azureml"
    else:
        mlflow.set_tracking_uri(f"file:{os.path.join(os.getcwd(),'mlruns')}")
        backend = "file"
    print(f"🧪 MLflow backend = {backend}")

    def pos_rate(df):
        return float(df["high_growth_label"].mean()) if len(df) else np.nan

    with mlflow.start_run():
        mlflow.log_params({
            "tickers": args.tickers,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "horizon_days": args.target_horizon_days,
            "high_growth_threshold": args.high_growth_threshold,
            "drop_outlier_flags": args.drop_outlier_flags,
            "price_col": price_col,
        })
        mlflow.log_metrics({
            "rows_total": len(data),
            "rows_train": len(train_df),
            "rows_val": len(val_df),
            "rows_test": len(test_df),
            "pos_rate_train": pos_rate(train_df),
            "pos_rate_val": pos_rate(val_df),
            "pos_rate_test": pos_rate(test_df),
            "n_features": len(feature_cols),
        })
        mlflow.log_artifact(feat_csv_local)
except Exception as e:
    print(f"⚠️ MLflow logging skipped: {e}")

print("✅ Training data preparation complete.")
