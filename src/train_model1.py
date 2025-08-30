# Phase 5 - Model 1 (FAST)
import os, io, time, argparse, joblib, warnings, inspect
import numpy as np, pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score, precision_recall_curve
)
from xgboost import XGBClassifier
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

# ---------------- args ----------------
p = argparse.ArgumentParser()
p.add_argument("--vault_url", required=True)
p.add_argument("--secret_name", default="AzureBlobConnStr")
p.add_argument("--container", required=True)
p.add_argument("--train_blob", required=True)
p.add_argument("--val_blob", required=True)
p.add_argument("--test_blob", required=True)
p.add_argument("--sample_frac", type=float, default=0.02)   # FAST: 2%
p.add_argument("--n_iter", type=int, default=4)             # FAST: 4 candidates
p.add_argument("--cv", type=int, default=2)                 # FAST: 2-fold
p.add_argument("--optimize_for", default="f1", choices=["f1","recall","precision"])
p.add_argument("--test_eval_frac", type=float, default=0.2) # FAST: subset eval
p.add_argument("--register", type=lambda s: s.lower()=="true", default=False)
p.add_argument("--upload_predictions", type=lambda s: s.lower()=="true", default=False)
args = p.parse_args()

# ---------------- KV + Blob ----------------
print("🔑 Using Key Vault -> Blob connection string")
cred = DefaultAzureCredential()
conn = SecretClient(vault_url=args.vault_url, credential=cred).get_secret(args.secret_name).value
bs = BlobServiceClient.from_connection_string(conn)
cc = bs.get_container_client(args.container)

def read_parquet_blob(path: str) -> pd.DataFrame:
    print(f"📥 {path}")
    b = cc.download_blob(path).readall()
    return pd.read_parquet(io.BytesIO(b))

# early permission check
cc.upload_blob("phase5_write_test.txt", b"ok", overwrite=True)
print("✅ Blob write test passed")

# ---------------- data ----------------
train_df = read_parquet_blob(args.train_blob)
val_df   = read_parquet_blob(args.val_blob)
test_df  = read_parquet_blob(args.test_blob)

def downcast(df):
    for c in df.select_dtypes("float"):
        df[c] = df[c].astype(np.float32)
    for c in df.select_dtypes("int"):
        df[c] = df[c].astype(np.int32)
    return df
train_df, val_df, test_df = map(downcast, [train_df, val_df, test_df])

# sample for search
train_sample = train_df.sample(frac=args.sample_frac, random_state=42)
exclude = {"Date","Ticker","future_return","high_growth_label"}
X_train, y_train = train_sample.drop(columns=list(exclude)), train_sample["high_growth_label"]
X_val,   y_val   = val_df.drop(columns=list(exclude)),       val_df["high_growth_label"]

# helper: check if this XGBoost version supports a fit() kwarg
def fit_supports(estimator, param_name: str) -> bool:
    try:
        return param_name in inspect.signature(estimator.fit).parameters
    except Exception:
        return False

# ---------------- search ----------------
param_dist = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.85, 1.0],
    "colsample_bytree": [0.85, 1.0],
    "min_child_weight": [1, 3],
    "gamma": [0, 0.1],
    "reg_lambda": [1, 1.5],
    "reg_alpha": [0, 0.1],
    "scale_pos_weight": [1, float(max(1, (y_train == 0).sum() / max(1, (y_train == 1).sum())))],
}

xgb = XGBClassifier(
    objective="binary:logistic",
    tree_method="hist",
    eval_metric="auc",
    n_jobs=-1,
    random_state=42,
)

print(f"🔍 RandomizedSearchCV on {len(train_sample):,} rows | n_iter={args.n_iter} | cv={args.cv}")
t0 = time.time()
warnings.filterwarnings("ignore", message="`early_stopping_rounds`")

# pass early stopping ONLY if this runtime supports it
search_fit_params = {}
if fit_supports(xgb, "eval_set"):
    search_fit_params["eval_set"] = [(X_val, y_val)]
if fit_supports(xgb, "early_stopping_rounds"):
    search_fit_params["early_stopping_rounds"] = 20

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=args.n_iter,
    scoring="roc_auc",
    n_jobs=-1,
    cv=args.cv,
    verbose=1,
    random_state=42
)
search.fit(X_train, y_train, **search_fit_params)
print(f"⏱ Tuning took {time.time()-t0:.1f}s")
print("✅ Best params:", search.best_params_, "| Best CV AUC:", round(search.best_score_, 4))

# ---------------- final fit ----------------
full_X = pd.concat([train_df.drop(columns=list(exclude)), val_df.drop(columns=list(exclude))], axis=0)
full_y = pd.concat([train_df["high_growth_label"], val_df["high_growth_label"]], axis=0)

final = XGBClassifier(
    **search.best_params_,
    objective="binary:logistic",
    tree_method="hist",
    eval_metric="auc",
    n_jobs=-1,
    random_state=42,
)

print("🚀 Training final model on full train+val")
final_fit_params = {}
if fit_supports(final, "eval_set"):
    final_fit_params["eval_set"] = [(X_val, y_val)]
if fit_supports(final, "early_stopping_rounds"):
    final_fit_params["early_stopping_rounds"] = 20

final.fit(full_X, full_y, **final_fit_params)

# ---------------- threshold on val ----------------
val_probs = final.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, val_probs)
if args.optimize_for == "f1":
    f1s = (2 * prec * rec) / (prec + rec + 1e-8); k = np.argmax(f1s)
elif args.optimize_for == "recall":
    k = np.argmax(rec)
else:
    k = np.argmax(prec)
best_thr = thr[max(0, min(k, len(thr) - 1))]
print(f"🎯 Best {args.optimize_for} threshold = {best_thr:.3f}")

# ---------------- test eval (subset for speed) ----------------
test_part = test_df.sample(frac=args.test_eval_frac, random_state=42) if 0 < args.test_eval_frac < 1 else test_df
X_test = test_part.drop(columns=list(exclude))
y_test = test_part["high_growth_label"]
probs  = final.predict_proba(X_test)[:, 1]
preds  = (probs >= best_thr).astype(int)
print("\n📈 TEST SAMPLE RESULTS")
print(classification_report(y_test, preds, digits=3))
print("ROC AUC:", round(roc_auc_score(y_test, probs), 4))

# ---------------- MLflow (Azure-safe) ----------------
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
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("test_auc", float(roc_auc_score(y_test, probs)))
        mlflow.log_metric("best_threshold", float(best_thr))
        mlflow.log_metric("precision", float(precision_score(y_test, preds)))
        mlflow.log_metric("recall", float(recall_score(y_test, preds)))
        mlflow.log_metric("f1", float(f1_score(y_test, preds)))
        mlflow.xgboost.log_model(final, artifact_path="model")
        if args.register and backend == "azureml":
            try:
                mlflow.xgboost.log_model(
                    final,
                    artifact_path="model_registry",
                    registered_model_name="HighGrowthStock_Model1",
                )
                print("✅ Model registered in Azure ML Registry")
            except Exception as e:
                print(f"⚠️ Registry step skipped: {e}")
except Exception as e:
    print(f"⚠️ MLflow logging skipped: {e}")

# ---------------- save model ----------------
joblib.dump(final, "model1_xgb.pkl")
with open("model1_xgb.pkl", "rb") as f:
    cc.upload_blob("model1/model1_xgb.pkl", f, overwrite=True)
print("☁️ Uploaded model1/model1_xgb.pkl")

# ---------------- split-wise predictions for Phase-6 ----------------
def save_split_preds(df: pd.DataFrame, split: str):
    Xs = df.drop(columns=list(exclude))
    ps = final.predict_proba(Xs)[:, 1]
    out = df[["Date","Ticker","future_return","high_growth_label"]].copy()
    out["model1_prob"] = ps
    out["model1_pred"] = (ps >= best_thr).astype(int)
    buf = io.BytesIO(); out.to_parquet(buf, index=False); buf.seek(0)
    cc.upload_blob(f"predictions/model1_predictions_{split}.parquet", buf.getvalue(), overwrite=True)
    print(f"☁️ predictions/model1_predictions_{split}.parquet")

if args.upload_predictions:
    save_split_preds(train_df, "train")
    save_split_preds(val_df,   "val")
    save_split_preds(test_df,  "test")   # full test, not just subset

print("🎉 Phase 5 FAST run completed successfully.")
