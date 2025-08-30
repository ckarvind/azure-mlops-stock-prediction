# ========================
# Phase-1: US Stock Data Ingestion (Prod, robust AML init + MLflow fallback)
# ========================
import argparse, os, time, sys
from datetime import datetime
import pandas as pd
import yfinance as yf
import requests
from tqdm import tqdm
from io import BytesIO

# Azure SDKs
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

import mlflow


# --------------------------
# Helper: robust MLClient init (works in AML jobs & locally)
# --------------------------
def get_ml_client(cred):
    # Prefer AML job env vars if present
    sub = (os.getenv("AZUREML_ARM_SUBSCRIPTION")
           or os.getenv("AZUREML_SUBSCRIPTION_ID")
           or os.getenv("AZ_SUBSCRIPTION_ID"))
    rg  = (os.getenv("AZUREML_ARM_RESOURCEGROUP")
           or os.getenv("AZUREML_RESOURCE_GROUP")
           or os.getenv("AZ_RESOURCE_GROUP"))
    ws  = (os.getenv("AZUREML_ARM_WORKSPACE_NAME")
           or os.getenv("AZUREML_WORKSPACE_NAME")
           or os.getenv("AZ_ML_WORKSPACE"))
    if sub and rg and ws:
        try:
            return MLClient(cred, subscription_id=sub, resource_group_name=rg, workspace_name=ws)
        except Exception as e:
            print(f"⚠️ MLClient via AML env vars failed: {e}")

    # Try explicit config dir if provided
    cfg_dir = os.getenv("AZUREML_CONFIG_DIR", ".")
    cfg_path = os.path.join(cfg_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            return MLClient.from_config(credential=cred, path=cfg_dir)
        except Exception as e:
            print(f"⚠️ MLClient.from_config(path={cfg_dir}) failed: {e}")

    # Fallback to default search (local dev)
    try:
        return MLClient.from_config(credential=cred)
    except Exception as e:
        print(f"⚠️ MLClient.from_config() not available: {e}")
        return None


# --------------------------
# Ticker fetch (scrape or fallback to Blob CSV)
# --------------------------
def get_wiki_tickers(url, col_keyword):
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        for df in tables:
            for col in df.columns:
                if col_keyword in str(col):
                    return df[col].astype(str).str.replace(".", "-", regex=False).tolist()
    except Exception:
        return []
    return []

def get_all_tickers(args, blob_service_client):
    sp500  = get_wiki_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol")
    nasdaq = get_wiki_tickers("https://en.wikipedia.org/wiki/NASDAQ-100", "Ticker")
    dow    = get_wiki_tickers("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", "Symbol")
    all_tickers = sorted(set(sp500 + nasdaq + dow))

    if len(all_tickers) > 100:
        print(f"✅ Loaded {len(all_tickers)} tickers from web")
        return all_tickers

    print("⚠️ Web scraping failed, falling back to CSV in Blob...")
    container_client = blob_service_client.get_container_client(args.container)
    blobs = list(container_client.list_blobs())
    csv_candidates = [b.name for b in blobs if "qualified_us_tickers" in b.name]
    if not csv_candidates:
        raise RuntimeError("No qualified_us_tickers CSV found in Blob fallback.")
    latest_csv = sorted(csv_candidates)[-1]
    print(f"📂 Using fallback tickers from {latest_csv}")

    blob_client = container_client.get_blob_client(latest_csv)
    csv_data = blob_client.download_blob().readall()
    tickers = pd.read_csv(BytesIO(csv_data)).iloc[:, 0].tolist()
    print(f"✅ Loaded {len(tickers)} tickers from Blob CSV")
    return tickers


# --------------------------
# Validate stock dataframe
# --------------------------
def validate_stock_data(df, cutoff_date):
    required_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Date"}
    if not required_cols.issubset(df.columns):
        return False
    if df.empty or df["Date"].min() > cutoff_date:
        return False
    return True


# --------------------------
# Fetch + Save single ticker
# --------------------------
def fetch_and_save(ticker, args, blob_service_client, cutoff_date, version_tag):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="max", auto_adjust=False)
        if df.empty:
            return ticker, "skip"

        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        df["Ticker"] = ticker
        df = df.dropna()

        if not validate_stock_data(df, cutoff_date):
            return ticker, "skip"

        local_path = f"{args.output_prefix}/{ticker}_{version_tag}.parquet"
        os.makedirs(args.output_prefix, exist_ok=True)
        df.to_parquet(local_path, index=False)

        blob_path = f"{args.blob_folder}{ticker}_{version_tag}.parquet"
        container_client = blob_service_client.get_container_client(args.container)
        with open(local_path, "rb") as data:
            container_client.upload_blob(blob_path, data, overwrite=True)

        return ticker, "ok"
    except Exception as e:
        print(f"⚠️ {ticker}: {e}")
        return ticker, "skip"   # never crash on a single ticker


# --------------------------
# Register dataset in AML (best-effort)
# --------------------------
def register_dataset(ml_client, args, version_tag):
    if ml_client is None:
        print("⚠️ Skipping AML dataset registration (no workspace context found).")
        return
    data_asset = Data(
        path=f"azureml://datastores/workspaceblobstore/paths/{args.container}/{args.blob_folder}",
        type=AssetTypes.URI_FOLDER,
        description=f"US stock data parquet (Adj Close, {args.min_years}+ yrs history)",
        name=f"us_stock_data_adjclose_{version_tag}"
    )
    ml_client.data.create_or_update(data_asset)
    print(f"✅ Registered dataset: {data_asset.name}")


# --------------------------
# Main
# --------------------------
def main(args):
    start_time = time.time()
    cred = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=args.vault_url, credential=cred)
    AZURE_CONN_STR = secret_client.get_secret(args.secret_name).value
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)

    ml_client = get_ml_client(cred)  # may be None; we’ll skip registration in that case

    version_tag = datetime.now().strftime("%Y%m%d")
    cutoff_date = pd.Timestamp(datetime.now() - pd.DateOffset(years=args.min_years)).tz_localize("UTC")

    ok_count, skip_count = 0, 0
    tickers_used = []

    if args.refresh.lower() == "true":
        tickers = get_all_tickers(args, blob_service_client)
        for i, t in enumerate(tqdm(tickers), 1):
            _, status = fetch_and_save(t, args, blob_service_client, cutoff_date, version_tag)
            if status == "ok":
                ok_count += 1
                tickers_used.append(t)
            else:
                skip_count += 1
            if i % 25 == 0:
                print(f"[{i}/{len(tickers)}] ok={ok_count}, skipped={skip_count}")
        print(f"\n✅ Completed refresh: {ok_count} ok, {skip_count} skipped")
    else:
        print("📂 Refresh disabled: skipping downloads")
        container_client = blob_service_client.get_container_client(args.container)
        blobs = list(container_client.list_blobs(name_starts_with=args.blob_folder))
        parquet_files = [b.name for b in blobs if b.name.endswith(".parquet")]
        tickers_used = [os.path.basename(p).split("_")[0] for p in parquet_files]
        print(f"📂 Found {len(tickers_used)} existing parquet files in {args.blob_folder}")

    # Try to register dataset (don’t fail the job if not possible)
    register_dataset(ml_client, args, version_tag)

    # --- MLflow logging (AzureML first, then fallback to local) ---
    tried_azureml = False
    try:
        if os.getenv("AZUREML_RUN_ID"):
            tried_azureml = True
            mlflow.set_tracking_uri("azureml://")
        else:
            mlflow.set_tracking_uri(f"file:{os.getcwd()}/mlruns")

        with mlflow.start_run():
            mlflow.log_params({
                "vault_url": args.vault_url,
                "container": args.container,
                "blob_folder": args.blob_folder,
                "output_prefix": args.output_prefix,
                "min_years": args.min_years,
                "refresh": args.refresh
            })
            mlflow.log_metrics({
                "ok_count": ok_count,
                "skip_count": skip_count,
                "elapsed_sec": time.time() - start_time
            })
            if tickers_used:
                tickers_out = os.path.join(args.output_prefix, f"tickers_{version_tag}.csv")
                os.makedirs(args.output_prefix, exist_ok=True)
                pd.DataFrame({"ticker": tickers_used}).to_csv(tickers_out, index=False)
                mlflow.log_artifact(tickers_out, artifact_path="tickers")
    except Exception as e:
        print(f"⚠️ MLflow tracking failed ({'azureml' if tried_azureml else 'local'}): {e}")
        # Fallback: local file store
        try:
            mlflow.set_tracking_uri(f"file:{os.getcwd()}/mlruns")
            with mlflow.start_run():
                mlflow.log_params({
                    "vault_url": args.vault_url,
                    "container": args.container,
                    "blob_folder": args.blob_folder,
                    "output_prefix": args.output_prefix,
                    "min_years": args.min_years,
                    "refresh": args.refresh
                })
                mlflow.log_metrics({
                    "ok_count": ok_count,
                    "skip_count": skip_count,
                    "elapsed_sec": time.time() - start_time
                })
        except Exception as e2:
            print(f"⚠️ MLflow logging skipped due to secondary failure: {e2}")

    # Always succeed (per your gating pattern, failures are handled internally)
    sys.exit(0)


# --------------------------
# CLI args
# --------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault_url", required=True)
    ap.add_argument("--secret_name", required=True)
    ap.add_argument("--container", required=True)
    ap.add_argument("--blob_folder", required=True)
    ap.add_argument("--output_prefix", default="stockdata_us_adjclose")
    ap.add_argument("--min_years", type=int, default=20)
    ap.add_argument("--refresh", default="false")
    args = ap.parse_args()
    main(args)
