# Code

This folder contains the core Python scripts for the AlphaEdge 
multimodal stock price prediction system developed as part of the 
dissertation:

**Multimodal Stock Price Prediction Leveraging Insider Transactions 
and Market Sentiment**  
Stella Umunna — George Washington University SEAS, DEng in AI, 2026

## Scripts

| File | Description |
|------|-------------|
| `fetch_form4_sec.py` | Fetches SEC Form 4 insider transaction filings via EDGAR |
| `fetch_sentiment.py` | Retrieves and scores daily news sentiment using FinBERT |
| `build_prices_cache.py` | Downloads and caches historical price data for S&P 500 and S&P 600 universe |
| `build_macro_features.py` | Constructs macroeconomic features including VIX, DXY, CPI, and FOMC signals |
| `features_insider.py` | Engineers insider transaction features from raw Form 4 data |
| `feature_pipeline.py` | Orchestrates the full feature pipeline across all 130 features and 6 feature groups |
| `train_regime_gated_model.py` | Trains the regime-gated specialist ensemble (Transformer + LightGBM + isotonic calibration) |
| `advisory_dashboard.py` | Generates the weekly advisory output with regime scores and directional signals |

## Requirements

Python 3.10+. Key dependencies: `lightgbm`, `transformers`, `torch`, 
`pandas`, `yfinance`, `sec-edgar-downloader`, `shap`, `scikit-learn`.

## Usage

Run scripts in this order:
1. `fetch_form4_sec.py`
2. `fetch_sentiment.py`
3. `build_prices_cache.py`
4. `build_macro_features.py`
5. `features_insider.py`
6. `feature_pipeline.py`
7. `train_regime_gated_model.py`
8. `advisory_dashboard.py`
