# Phase 6 - Model 2 Regressor (XGBRegressor) with KV+Blob and MLflow
import os, io, time, argparse, warnings
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

# ------------- args -------------
p = argparse.ArgumentParser()
p.add_argument("--vault_url", required=True)
p.add_argument("--secret_name", default="AzureBlobConnStr")
p.add_argument("--container", required=True)
p.add_argument("--features_path", default="training_data")
p.add_argument("--predictions_path", default="predictions")
p.add_argument("--model1_threshold", type=float, default=0.20)
p.add_argument("--n_jobs", type=int, default=-1)
p.add_argument("--register", type=lambda s: s.lower()=="true", default=False)
args = p.parse_args()

# ------------- KV + Blob -------------
cred = DefaultAzureCredential()
conn = SecretClient(vault_url=args.vault_url, credential=cred).get_secret(args.secret_name).value
bs = BlobServiceClient.from_connection_string(conn)
cc = bs.get_container_client(args.container)

from azure.core.exceptions import ResourceModifiedError, ResourceNotFoundError, HttpResponseError

def load_parquet(blob_path: str, attempts: int = 8) -> pd.DataFrame:
    for i in range(attempts):
        try:
            print(f"📥 {blob_path}")
            downloader = cc.download_blob(blob_path, max_concurrency=1)  # single-threaded
            b = downloader.readall()
            return pd.read_parquet(io.BytesIO(b))
        except ResourceModifiedError:
            wait = min(2 ** i, 30)
            print(f"⚠️ Blob modified during read. Retry {i+1}/{attempts} in {wait}s")
            time.sleep(wait)
        except HttpResponseError as e:
            wait = min(2 ** i, 30)
            print(f"⚠️ Storage error ({e}). Retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to read {blob_path} after {attempts} attempts")


def blob_exists(path: str) -> bool:
    try:
        cc.get_blob_client(path).get_blob_properties()
        return True
    except Exception:
        return False

# ------------- load features + m1 preds -------------
splits = ["train", "val", "test"]
features, m1preds = {}, {}
for sp in splits:
    fpath = f"{args.features_path}/{sp}.parquet"
    ppath = f"{args.predictions_path}/model1_predictions_{sp}.parquet"
    if not (blob_exists(fpath) and blob_exists(ppath)):
        raise FileNotFoundError(f"Missing required blob(s) for split={sp}: {fpath} or {ppath}")
    features[sp] = load_parquet(fpath)
    m1preds[sp]  = load_parquet(ppath)
    print(f"✅ {sp.upper()} features: {features[sp].shape}, m1 preds: {m1preds[sp].shape}")

# ------------- merge + candidate filter -------------
candidates = {}
for sp in splits:
    # bring in ONLY what we need from model1 to avoid _x/_y label duplicates
    keep = ["Date", "Ticker", "model1_prob", "model1_pred"]
    m1 = m1preds[sp][[c for c in keep if c in m1preds[sp].columns]].copy()

    df = features[sp].merge(m1, on=["Date", "Ticker"], how="left")

    # be defensive: ensure numeric prob and fill any missing
    df["model1_prob"] = pd.to_numeric(df["model1_prob"], errors="coerce").fillna(0.0)

    before = len(df)
    df = df[df["model1_prob"] >= args.model1_threshold].reset_index(drop=True)
    after = len(df)
    print(f"🎯 {sp.upper()} candidates >= {args.model1_threshold}: {after:,} / {before:,}")
    candidates[sp] = df

if len(candidates["train"]) == 0 or len(candidates["val"]) == 0:
    raise RuntimeError("No candidates after filtering — lower --model1_threshold or check Model-1 outputs.")

exclude = {"Date","Ticker","future_return","high_growth_label","model1_prob","model1_pred"}
X_train = candidates["train"].drop(columns=list(exclude))
y_train = candidates["train"]["future_return"]
X_val   = candidates["val"].drop(columns=list(exclude))
y_val   = candidates["val"]["future_return"]


# ------------- train model2 -------------
warnings.filterwarnings("ignore", message="early_stopping_rounds")
print("🚀 Training Model 2 (XGBRegressor)")
t0 = time.time()
model2 = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.5,
    reg_alpha=0.1,
    n_jobs=args.n_jobs,
    random_state=42,
    tree_method="hist",
)
model2.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
print(f"⏱ Training done in {time.time()-t0:.1f}s")

# ------------- evaluate -------------
def eval_split(split: str):
    if len(candidates[split]) == 0:
        return np.nan, np.nan
    Xs = candidates[split].drop(columns=list(exclude))
    ys = candidates[split]["future_return"]
    pr = model2.predict(Xs)
    rmse = mean_squared_error(ys, pr, squared=False)
    r2   = r2_score(ys, pr)
    print(f"📊 {split.upper()} RMSE={rmse:.4f}, R²={r2:.4f}")
    return rmse, r2

rmse_val, r2_val = eval_split("val")
rmse_test, r2_test = eval_split("test")

# ------------- MLflow logging + optional registry -------------
try:
    import mlflow, mlflow.xgboost
    if os.getenv("MLFLOW_TRACKING_URI", "").startswith("azureml://") or "AZUREML_RUN_ID" in os.environ:
        backend = "azureml"
    else:
        local_dir = os.path.join(os.getcwd(), "mlruns")
        mlflow.set_tracking_uri(f"file:{local_dir}")
        backend = "file"
    print(f"🧪 MLflow backend = {backend}")

    with mlflow.start_run():
        mlflow.log_param("model1_threshold", args.model1_threshold)
        mlflow.log_params({
            "n_estimators": 600, "learning_rate": 0.05, "max_depth": 6,
            "subsample": 0.85, "colsample_bytree": 0.85,
            "reg_lambda": 1.5, "reg_alpha": 0.1
        })
        if not np.isnan(rmse_val):
            mlflow.log_metric("rmse_val", float(rmse_val))
            mlflow.log_metric("r2_val",   float(r2_val))
        if not np.isnan(rmse_test):
            mlflow.log_metric("rmse_test", float(rmse_test))
            mlflow.log_metric("r2_test",   float(r2_test))
        mlflow.xgboost.log_model(model2, artifact_path="model2")

        if args.register and backend == "azureml":
            try:
                mlflow.xgboost.log_model(
                    model2,
                    artifact_path="model2_registry",
                    registered_model_name="HighGrowthStock_Model2",
                )
                print("✅ Model-2 registered in Azure ML Registry")
            except Exception as e:
                print(f"⚠️ Model-2 registry step skipped: {e}")
except Exception as e:
    print(f"⚠️ MLflow logging skipped: {e}")

# ------------- persist model -------------
import joblib
joblib.dump(model2, "model2_xgb.pkl")
with open("model2_xgb.pkl", "rb") as f:
    cc.upload_blob("model2/model2_xgb.pkl", f, overwrite=True)
print("☁️ Uploaded model2/model2_xgb.pkl")

# ------------- write per-split predictions -------------
def save_split_preds(df: pd.DataFrame, split: str):
    if len(df) == 0:
        print(f"⚠️ No rows to save for split={split}")
        return
    Xs = df.drop(columns=list(exclude))
    pr = model2.predict(Xs)
    out = df[["Date","Ticker","future_return","high_growth_label","model1_prob","model1_pred"]].copy()
    out["model2_pred_return"] = pr
    buf = io.BytesIO()
    out.to_parquet(buf, index=False); buf.seek(0)
    cc.upload_blob(f"{args.predictions_path}/model2_predictions_{split}.parquet", buf.getvalue(), overwrite=True)
    print(f"☁️ {args.predictions_path}/model2_predictions_{split}.parquet")

for sp in splits:
    save_split_preds(candidates[sp], sp)

print("✅ Phase 6 complete.")
