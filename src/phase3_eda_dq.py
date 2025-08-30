#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 — EDA + Data Quality (production)
- Reads per-ticker parquet from <container>/<input_prefix>/
- Computes per-ticker + global stats with DQ thresholds
- Uploads CSV/JSON, interactive Plotly HTML (index.html), and PDF report (EDA_Report.pdf) to Blob:
    <output_prefix>/<version_ts>/
- Exits 0 (PASS) / 2 (FAIL)
"""

import os, io, sys, json, argparse, logging, time, tempfile
from io import BytesIO
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Headless plotting for PDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PDF
from fpdf import FPDF

# Plotly for interactive HTML
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Azure
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import HttpResponseError, ResourceModifiedError


# ---------------- CLI ----------------
p = argparse.ArgumentParser("Phase 3 — EDA + DQ")
p.add_argument("--vault_url", required=True)
p.add_argument("--secret_name", default="AzureBlobConnStr")
p.add_argument("--container", required=True)

p.add_argument("--input_prefix", default="model_ready_data",
               help="Blob folder with per-ticker parquet files")
p.add_argument("--output_prefix", default="eda/phase3",
               help="Blob folder to write EDA artifacts")
p.add_argument("--version_ts", default=datetime.utcnow().strftime("%Y%m%d"),
               help="Suffix folder for outputs (e.g., 20250827)")

# DQ thresholds
p.add_argument("--min_rows_per_ticker", type=int, default=750)
p.add_argument("--max_na_fraction", type=float, default=0.05)
p.add_argument("--max_duplicate_dates", type=int, default=0)
p.add_argument("--max_outlier_fraction", type=float, default=0.20)
p.add_argument("--min_pass_rate", type=float, default=0.95)
p.add_argument("--allow_missing_feature_cols", type=int, default=2)

# perf/logs
p.add_argument("--workers", type=int, default=min(cpu_count(), 12))
p.add_argument("--log_level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
args = p.parse_args()

# NEW: fallback if component passes empty string
if not args.version_ts:
    args.version_ts = datetime.utcnow().strftime("%Y%m%d")

logging.basicConfig(level=getattr(logging, args.log_level),
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("phase3")

# ---------------- Azure setup ----------------
log.info("🔑 Fetching Blob connection string from Key Vault")
cred = DefaultAzureCredential()
conn_str = SecretClient(vault_url=args.vault_url, credential=cred).get_secret(args.secret_name).value
bs = BlobServiceClient.from_connection_string(conn_str)
cc = bs.get_container_client(args.container)

def list_parquet(prefix: str) -> List[str]:
    prefix = prefix.strip("/") + "/"
    return [b.name for b in cc.list_blobs(name_starts_with=prefix) if b.name.lower().endswith(".parquet")]

def dl(path: str, attempts: int = 6) -> bytes:
    for i in range(attempts):
        try:
            return cc.download_blob(path, max_concurrency=1).readall()
        except (HttpResponseError, ResourceModifiedError) as e:
            wait = min(2 ** i, 30)
            log.warning(f"retry {i+1}/{attempts} for {path} in {wait}s ({e})")
            time.sleep(wait)
    raise RuntimeError(f"Failed to read {path} after {attempts} attempts")

def ul_bytes(data: bytes, path: str):
    cc.upload_blob(name=path, data=data, overwrite=True)
    log.info(f"☁️ uploaded: {path}")

def ul_df_csv(df: pd.DataFrame, path: str):
    b = io.BytesIO()
    df.to_csv(b, index=False)
    b.seek(0)
    ul_bytes(b.getvalue(), path)

def ul_json(obj: dict, path: str):
    b = io.BytesIO(json.dumps(obj, indent=2).encode("utf-8"))
    ul_bytes(b.getvalue(), path)


# ---------------- EDA/DQ core ----------------
REQUIRED_FEATURE_COLS = [
    "SMA_20","EMA_20","RSI_14","MACD","MACD_signal","BB_upper","BB_lower",
    "ATR_14","ADX_14","daily_return","volatility_20d"
]
REQUIRED_KEYS = ["Date","Ticker","Sector"]

def analyze_one(blob_name: str) -> Dict:
    try:
        df = pd.read_parquet(BytesIO(dl(blob_name)))
        if df.empty:
            return {
                "ticker":"UNKNOWN","sector":"Unknown","blob":blob_name,
                "rows":0,"cols":0,"date_min":None,"date_max":None,"dup_dates":None,
                "missing_keys":"", "missing_feats":"", "na_frac_feats":1.0, "outlier_frac_rows":1.0,
                "passed":False, "reasons":"EMPTY_DF"
            }

        # normalize
        if "Date" in df.columns and not np.issubdtype(df["Date"].dtype, np.datetime64):
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        ticker = str(df["Ticker"].iloc[0]) if "Ticker" in df.columns else "UNKNOWN"
        sector = str(df["Sector"].iloc[0]) if "Sector" in df.columns else "Unknown"

        n_rows = int(len(df)); n_cols = int(df.shape[1])
        dup_dates = int(df.duplicated(subset=["Ticker","Date"]).sum()) if {"Ticker","Date"}.issubset(df.columns) else None
        date_min = str(pd.to_datetime(df["Date"]).min().date()) if "Date" in df.columns else None
        date_max = str(pd.to_datetime(df["Date"]).max().date()) if "Date" in df.columns else None

        # coverage
        missing_keys = [k for k in REQUIRED_KEYS if k not in df.columns]
        missing_feats = [c for c in REQUIRED_FEATURE_COLS if c not in df.columns]
        feats_present = [c for c in REQUIRED_FEATURE_COLS if c in df.columns]
        na_frac = float(df[feats_present].isna().mean().mean()) if feats_present else 1.0

        outlier_cols = [c for c in df.columns if c.endswith("_is_outlier")]
        outlier_frac = float((df[outlier_cols].sum(axis=1) > 0).mean()) if outlier_cols else 0.0

        reasons = []
        if missing_keys: reasons.append(f"missing_keys={missing_keys}")
        if n_rows < args.min_rows_per_ticker: reasons.append(f"rows<{args.min_rows_per_ticker}")
        if dup_dates is not None and dup_dates > args.max_duplicate_dates: reasons.append(f"dup_dates>{args.max_duplicate_dates}")
        if len(missing_feats) > args.allow_missing_feature_cols: reasons.append(f"missing_feats>{args.allow_missing_feature_cols}")
        if na_frac > args.max_na_fraction: reasons.append(f"na_frac>{args.max_na_fraction}")
        if outlier_frac > args.max_outlier_fraction: reasons.append(f"outlier_frac>{args.max_outlier_fraction}")
        passed = len(reasons) == 0

        return {
            "ticker": ticker, "sector": sector, "blob": blob_name,
            "rows": n_rows, "cols": n_cols,
            "date_min": date_min, "date_max": date_max,
            "dup_dates": dup_dates,
            "missing_keys": ",".join(missing_keys) if missing_keys else "",
            "missing_feats": ",".join(missing_feats) if missing_feats else "",
            "na_frac_feats": round(na_frac, 6),
            "outlier_frac_rows": round(outlier_frac, 6),
            "passed": passed,
            "reasons": ";".join(reasons),
        }
    except Exception as e:
        return {
            "ticker":"UNKNOWN","sector":"Unknown","blob":blob_name,
            "rows":0,"cols":0,"date_min":None,"date_max":None,"dup_dates":None,
            "missing_keys":"", "missing_feats":"", "na_frac_feats":1.0, "outlier_frac_rows":1.0,
            "passed":False,"reasons":f"EXCEPTION:{type(e).__name__}:{e}"
        }


# discover files
blobs = list_parquet(args.input_prefix)
log.info(f"📂 Found {len(blobs)} parquet under {args.input_prefix}/")
t0 = time.time()

rows: List[Dict] = []
if args.workers and args.workers > 1:
    with Pool(processes=args.workers) as pool:
        for r in pool.imap_unordered(analyze_one, blobs):
            rows.append(r)
else:
    for b in blobs:
        rows.append(analyze_one(b))

eda_df = pd.DataFrame(rows).sort_values(["passed","ticker"]).reset_index(drop=True)
pass_rate = float(eda_df["passed"].mean()) if not eda_df.empty else 0.0

by_sector = eda_df.groupby("sector", dropna=False)["ticker"].nunique().reset_index(name="n_tickers")
na_stats = eda_df[["na_frac_feats","outlier_frac_rows"]].describe().T.reset_index().rename(columns={"index":"metric"})

summary = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "tickers_total": int(eda_df["ticker"].nunique()) if not eda_df.empty else 0,
    "tickers_pass": int(eda_df.loc[eda_df["passed"], "ticker"].nunique()) if not eda_df.empty else 0,
    "tickers_fail": int(eda_df.loc[~eda_df["passed"], "ticker"].nunique()) if not eda_df.empty else 0,
    "pass_rate": round(pass_rate, 6),
    "thresholds": {
        "min_rows_per_ticker": args.min_rows_per_ticker,
        "max_na_fraction": args.max_na_fraction,
        "max_duplicate_dates": args.max_duplicate_dates,
        "max_outlier_fraction": args.max_outlier_fraction,
        "allow_missing_feature_cols": args.allow_missing_feature_cols,
        "min_pass_rate": args.min_pass_rate
    }
}
status = "PASS" if pass_rate >= args.min_pass_rate else "FAIL"
log.info(f"⏱ Finished in {time.time()-t0:.1f}s | pass_rate={pass_rate:.3f} -> {status}")

# ---------------- Upload CSV/JSON ----------------
base = f"{args.output_prefix.strip('/')}/{args.version_ts}"
ul_df_csv(eda_df,           f"{base}/dq_summary_per_ticker.csv")
ul_df_csv(by_sector,        f"{base}/tickers_by_sector.csv")
ul_df_csv(na_stats,         f"{base}/global_na_outlier_stats.csv")
ul_json(summary,            f"{base}/global_metrics.json")
ul_json({"status": status, "summary": summary}, f"{base}/dq_status.json")

# ---------------- Interactive HTML (Plotly) ----------------
try:
    # PASS/FAIL bar
    pf_counts = eda_df['passed'].value_counts().rename({True:'PASS', False:'FAIL'}).reindex(['PASS','FAIL']).fillna(0)
    fig_passfail = px.bar(
        pf_counts.reset_index().rename(columns={'index':'status','passed':'count'}),
        x='status', y='count',
        title='Tickers PASS vs FAIL'
    )

    # Sector coverage (top 20)
    srt = by_sector.sort_values('n_tickers', ascending=False).head(20)
    fig_sector = px.bar(srt, x='n_tickers', y='sector', orientation='h',
                        title='Top Sectors by #Tickers')

    # NA / Outlier histograms
    fig_na = px.histogram(eda_df, x='na_frac_feats', nbins=40, title='NA Fraction (per ticker)')
    fig_out = px.histogram(eda_df, x='outlier_frac_rows', nbins=40, title='Outlier Row Fraction (per ticker)')

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=summary['pass_rate']*100,
        title={'text': "Pass Rate (%)"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'thickness': 0.3}}
    ))

    parts = []
    for f in (fig_gauge, fig_passfail, fig_sector, fig_na, fig_out):
        parts.append(pio.to_html(f, include_plotlyjs='cdn', full_html=False))

    header = f"""
    <h1>Phase 3 — EDA & Data Quality</h1>
    <p><b>Status</b>: <span style="color:{'#16a34a' if status=='PASS' else '#dc2626'}">{status}</span>
       | <b>Pass Rate</b>: {summary['pass_rate']:.3f}
       | <b>Tickers PASS/Total</b>: {summary['tickers_pass']}/{summary['tickers_total']}
       | <b>Generated</b>: {summary['generated_utc']}</p>
    <pre style="background:#f7f7f7;padding:8px;border:1px solid #ddd">
{json.dumps(summary['thresholds'], indent=2)}
    </pre>
    <p>Downloads:
      <a href="dq_summary_per_ticker.csv">dq_summary_per_ticker.csv</a> ·
      <a href="tickers_by_sector.csv">tickers_by_sector.csv</a> ·
      <a href="global_na_outlier_stats.csv">global_na_outlier_stats.csv</a> ·
      <a href="global_metrics.json">global_metrics.json</a> ·
      <a href="dq_status.json">dq_status.json</a> ·
      <a href="EDA_Report.pdf">EDA_Report.pdf</a>
    </p>
    """

    html = f"""<!doctype html><html><head><meta charset="utf-8">
    <title>Phase 3 — EDA & Data Quality</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    </head><body style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px">
    {header}
    {''.join(parts)}
    </body></html>"""

    ul_bytes(html.encode("utf-8"), f"{base}/index.html")
    log.info("📝 Uploaded interactive index.html")
except Exception as e:
    log.warning(f"Interactive HTML skipped: {e}")

# ---------------- PDF report ----------------
try:
    tmpdir = tempfile.mkdtemp(prefix="eda_plots_")

    # Plot 1: NA histogram
    plt.figure(figsize=(7.5,3.8))
    plt.hist(eda_df["na_frac_feats"].dropna().values, bins=40)
    plt.title("NA Fraction (per ticker)"); plt.xlabel("NA fraction"); plt.ylabel("Count")
    p1 = os.path.join(tmpdir, "na_hist.png")
    plt.tight_layout(); plt.savefig(p1, dpi=120, bbox_inches="tight"); plt.close()

    # Plot 2: Outlier histogram
    plt.figure(figsize=(7.5,3.8))
    plt.hist(eda_df["outlier_frac_rows"].dropna().values, bins=40)
    plt.title("Outlier Row Fraction (per ticker)"); plt.xlabel("Outlier fraction"); plt.ylabel("Count")
    p2 = os.path.join(tmpdir, "outlier_hist.png")
    plt.tight_layout(); plt.savefig(p2, dpi=120, bbox_inches="tight"); plt.close()

    # Plot 3: Sectors (top 15)
    s2 = by_sector.sort_values("n_tickers", ascending=False).head(15)
    plt.figure(figsize=(7.5,4.6))
    y = np.arange(len(s2))
    plt.barh(y, s2["n_tickers"].values)
    plt.yticks(y, s2["sector"].values)
    plt.title("Top Sectors by #Tickers")
    plt.xlabel("#Tickers")
    p3 = os.path.join(tmpdir, "sectors.png")
    plt.tight_layout(); plt.savefig(p3, dpi=120, bbox_inches="tight"); plt.close()

    # Build PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    def add_title(txt): pdf.set_font("Arial", "B", 14); pdf.cell(0, 10, txt, ln=True)
    def add_sub(txt):   pdf.set_font("Arial", "B", 11); pdf.cell(0, 7, txt, ln=True)
    def add_text(txt):  pdf.set_font("Arial", "", 10);  pdf.multi_cell(0, 5, txt); pdf.ln(1)

    pdf.add_page()
    add_title("Stock Data — Phase 3 EDA & Data Quality")
    add_text(f"Generated: {summary['generated_utc']}")
    add_text(f"Status: {status} | Pass Rate: {summary['pass_rate']:.3f} | Tickers PASS/Total: {summary['tickers_pass']}/{summary['tickers_total']}")

    add_sub("Thresholds")
    add_text(json.dumps(summary["thresholds"], indent=2))

    add_sub("Global NA/Outlier Stats")
    add_text(na_stats.to_string(index=False))

    # Images
    add_sub("NA Fraction (per ticker)")
    pdf.image(p1, w=180)
    add_sub("Outlier Row Fraction (per ticker)")
    pdf.image(p2, w=180)
    add_sub("Top Sectors by #Tickers")
    pdf.image(p3, w=180)

    # Fail reasons (top 20)
    bad = eda_df.loc[~eda_df["passed"], ["ticker","reasons"]].head(20)
    add_sub("Sample of Failed Tickers (top 20)")
    if bad.empty: add_text("None")
    else: add_text(bad.to_string(index=False))

    pdf_name = "EDA_Report.pdf"
    pdf.output(pdf_name)
    with open(pdf_name, "rb") as f:
        ul_bytes(f.read(), f"{base}/{pdf_name}")
    log.info("📄 Uploaded EDA_Report.pdf")
except Exception as e:
    log.warning(f"PDF generation skipped: {e}")

# ---------------- optional MLflow ----------------
try:
    import mlflow
    if os.getenv("MLFLOW_TRACKING_URI","").startswith("azureml://") or "AZUREML_RUN_ID" in os.environ:
        pass
    else:
        mlflow.set_tracking_uri(f"file:{os.path.join(os.getcwd(),'mlruns')}")
    with mlflow.start_run():
        mlflow.log_metric("dq_pass_rate", pass_rate)
        for k, v in summary["thresholds"].items():
            mlflow.log_param(f"dq_{k}", v)
        eda_df.head(1000).to_csv("dq_summary_sample.csv", index=False)
        mlflow.log_artifact("dq_summary_sample.csv")
        if os.path.exists("EDA_Report.pdf"):
            mlflow.log_artifact("EDA_Report.pdf")
        if os.path.exists("index.html"):
            mlflow.log_artifact("index.html")
except Exception as e:
    log.warning(f"MLflow logging skipped: {e}")

# ---------------- exit for gating ----------------
sys.exit(0 if status == "PASS" else 2)
