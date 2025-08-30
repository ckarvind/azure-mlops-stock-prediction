# azure-mlops-stock-prediction
This project uses yfinance, LSTM, and XGBoost with Azure Machine Learning.

## Overview
End-to-end Azure MLOps pipeline for stock market prediction and algorithmic trading.  
Covers data ingestion, feature engineering, model training, scoring, backtesting, and dashboarding.  

## Repo Structure
notebooks/    - Development notebooks  
src/          - Production-ready Python scripts (Phases 1–9)  
components/   - Azure ML component YAMLs  
config/       - Environments & pipeline YAMLs  
dashboards/   - Streamlit app (Phase 9)  

## Steps
1. Create Azure ML environment from config/env_phase5.yml  
2. Register components using `az ml component create`  
3. Run pipeline using `az ml job create --file config/pipeline_job.yml`  
4. Publish pipeline and trigger via ADF  
5. Deploy dashboard (Phase 9) via Azure App Service  

## Notes
- Secrets (Key Vault, connection strings) should be externalized, not hardcoded.  
- Data (raw OHLCV, model-ready, predictions, backtest) lives in Blob Storage.  

