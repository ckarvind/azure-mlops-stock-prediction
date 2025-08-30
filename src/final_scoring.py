# Phase 7 — Final Scoring & Top-N Selection (production)
import os, io, argparse, json
import numpy as np, pandas as pd

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

# ---------------- args ----------------
p = argparse.ArgumentParser()
p.add_argument("--vault_url", required=True)
p.add_argument("--secret_name", default="AzureBlobConnStr")
p.add_argument("--container", required=True)

p.add_argument("--predictions_path", default="predictions")          # where model1_*/model2_* live
p.add_argument("--output_prefix",   default="predictions/final_scored")
p.add_argument("--top_n", type=int, default=10)
p.add_argument("--splits", default="train,val,test")                 # comma-separated
p.add_argument("--skip_missing", type=lambda s: s.lower()=="true", default=True)

args = p.parse_args()

# ---------------- KV + Blob ----------------
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

def load_parquet(path: str) -> pd.DataFrame:
    print(f"📥 {path}")
    b = cc.download_blob(path).readall()
    return pd.read_parquet(io.BytesIO(b))

def upload_parquet(df: pd.DataFrame, path: str):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    cc.upload_blob(name=path, data=buf.getvalue(), overwrite=True)
    print(f"☁️ Uploaded: {path}")

# ---------------- utilities ----------------
def pick_col(cols, preferred, fallbacks):
    """Return first matching column from [preferred]+fallbacks via exact then substring search."""
    if preferred in cols:
        return preferred
    for pat in fallbacks:
        cands = [c for c in cols if pat in c]
        if cands:
            return cands[0]
    raise KeyError(f"Could not find one of { [preferred]+fallbacks } in columns: {list(cols)[:15]}...")

def standardize_keys(df: pd.DataFrame) -> pd.DataFrame:
    # normalize join keys to avoid dtype merge issues
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

def process_split(split: str, top_n: int) -> dict:
    """
    Returns metrics dict. Writes final_scored parquet to Blob.
    """
    m1_path = f"{args.predictions_path}/model1_predictions_{split}.parquet"
    m2_path = f"{args.predictions_path}/model2_predictions_{split}.parquet"

    if not (blob_exists(m1_path) and blob_exists(m2_path)):
        msg = f"⚠️  Skipping {split}: missing one of {m1_path} or {m2_path}"
        if args.skip_missing:
            print(msg)
            return {"split": split, "status": "skipped"}
        else:
            raise FileNotFoundError(msg)

    m1 = load_parquet(m1_path)
    m2 = load_parquet(m2_path)
    print(f"✅ {split.upper()} — Model1: {m1.shape}, Model2: {m2.shape}")

    # ensure keys clean
    m1 = standardize_keys(m1)
    m2 = standardize_keys(m2)

    # de-dup on (Date, Ticker) if any (keep last)
    if m1.duplicated(subset=["Date","Ticker"]).any():
        m1 = m1.sort_values(["Date","Ticker"]).drop_duplicates(["Date","Ticker"], keep="last")
    if m2.duplicated(subset=["Date","Ticker"]).any():
        m2 = m2.sort_values(["Date","Ticker"]).drop_duplicates(["Date","Ticker"], keep="last")

    # ---- find/rename score columns (robust to variants)
    prob_col_m1 = pick_col(
        m1.columns,
        preferred="model1_prob",
        fallbacks=["prob", "probability", "score"]
    )
    pred_col_m2 = pick_col(
        m2.columns,
        preferred="model2_pred_return",
        fallbacks=["pred_return", "model2_pred", "prediction", "pred"]
    )

    # keep small projections to avoid suffix chaos
    m1 = m1[["Date","Ticker", prob_col_m1] + (["model1_pred"] if "model1_pred" in m1.columns else [])]
    m1 = m1.rename(columns={prob_col_m1: "model1_prob_m1",
                            "model1_pred": "model1_pred_m1" if "model1_pred" in m1.columns else "model1_pred"})
    m2 = m2[["Date","Ticker", pred_col_m2] + (["future_return"] if "future_return" in m2.columns else [])]
    m2 = m2.rename(columns={pred_col_m2: "model2_pred_return_m2"})

    # ---- merge
    merged = m1.merge(m2, on=["Date","Ticker"], how="inner")
    if merged.empty:
        print(f"⚠️  {split.upper()} — empty after merge; nothing to rank.")
        return {"split": split, "status": "empty_merge"}

    # ---- compute final score & rank per date
    merged["final_score"] = merged["model1_prob_m1"] * merged["model2_pred_return_m2"]
    merged["rank"] = merged.groupby("Date")["final_score"].rank(ascending=False, method="first")
    topN = merged[merged["rank"] <= top_n].reset_index(drop=True)

    # ---- write parquet
    out_path = f"{args.output_prefix}_{split}.parquet"
    upload_parquet(topN, out_path)

    # ---- summarize metrics
    metrics = {
        "split": split,
        "status": "ok",
        "rows_input_m1": int(len(m1)),
        "rows_input_m2": int(len(m2)),
        "rows_merged": int(len(merged)),
        "rows_topN": int(len(topN)),
        "unique_dates": int(topN["Date"].nunique()),
        "top_n": int(top_n),
        "final_score_mean": float(np.nanmean(topN["final_score"])) if len(topN) else np.nan,
        "final_score_median": float(np.nanmedian(topN["final_score"])) if len(topN) else np.nan,
    }
    if "future_return" in topN.columns:
        metrics["future_return_mean"] = float(np.nanmean(topN["future_return"]))
        metrics["future_return_median"] = float(np.nanmedian(topN["future_return"]))
        # correlation can be informative; guard against NaNs
        try:
            corr = topN["final_score"].corr(topN["future_return"])
            metrics["corr_score_future_return"] = float(corr) if pd.notnull(corr) else np.nan
        except Exception:
            metrics["corr_score_future_return"] = np.nan

    return metrics

# ---------------- run all splits ----------------
splits = [s.strip() for s in args.splits.split(",") if s.strip()]
all_metrics = []

for sp in splits:
    m = process_split(sp, args.top_n)
    all_metrics.append(m)

# ---------------- MLflow logging & quick-look artifacts ----------------
try:
    import mlflow
    if os.getenv("MLFLOW_TRACKING_URI", "").startswith("azureml://") or "AZUREML_RUN_ID" in os.environ:
        backend = "azureml"
    else:
        mlflow.set_tracking_uri(f"file:{os.path.join(os.getcwd(),'mlruns')}")
        backend = "file"
    print(f"🧪 MLflow backend = {backend}")

    with mlflow.start_run():
        mlflow.log_param("top_n", int(args.top_n))
        mlflow.log_param("splits", ",".join(splits))
        mlflow.log_param("output_prefix", args.output_prefix)

        # Log per-split metrics
        for m in all_metrics:
            split = m.get("split", "unknown")
            for k, v in m.items():
                if k in ("split", "status"): 
                    continue
                if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                    mlflow.log_metric(f"{split}_{k}", v)

        # Persist metrics JSON as artifact
        meta_path = "final_scoring_metrics.json"
        with open(meta_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=lambda x: None)
        mlflow.log_artifact(meta_path)

        # Optional: quick-look "latest Top-10" for TEST split as artifact
        test_out = f"{args.output_prefix}_test.parquet"
        if blob_exists(test_out):
            df_test = load_parquet(test_out)
            if not df_test.empty and "Date" in df_test.columns:
                latest_date = df_test["Date"].max()
                top10_latest = (
                    df_test[df_test["Date"] == latest_date]
                    .sort_values("final_score", ascending=False)
                    .head(10)
                )
                # save local
                top10_csv = "top10_latest.csv"
                top10_parq = "top10_latest.parquet"
                top10_latest.to_csv(top10_csv, index=False)
                top10_latest.to_parquet(top10_parq, index=False)
                # log to MLflow
                mlflow.log_artifact(top10_csv)
                mlflow.log_artifact(top10_parq)
                # also upload to Blob for your dashboard convenience
                with open(top10_csv, "rb") as f:
                    cc.upload_blob("predictions/top10_latest.csv", f, overwrite=True)
                with open(top10_parq, "rb") as f:
                    cc.upload_blob("predictions/top10_latest.parquet", f, overwrite=True)
                print(f"☁️ Uploaded quick-look: predictions/top10_latest.(csv|parquet)")
except Exception as e:
    print(f"⚠️ MLflow logging skipped: {e}")

print("✅ Final scoring finished.")
