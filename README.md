Advanced Time Series Forecasting — README
Project overview

An end-to-end forecasting pipeline for S&P 500 daily closing prices focused on point forecasts and uncertainty quantification.
Main model: LSTM with Monte Carlo Dropout (produces predictive distributions and 90% prediction intervals). Includes classical baselines (ARIMA and Prophet) for comparison, calibration analysis, and exportable artifacts for serving.

What this repository contains

A single Google Colab–ready Python cell (or script) that runs the full pipeline:

Attempts to download S&P 500 (^GSPC) via yfinance; falls back to a synthetic S&P-like series if download fails.

Preprocessing: business-day frequency, forward-fill missing days, MinMaxScaler, sliding window creation.

Deep model: multi-step LSTM trained with dropout, MC Dropout at inference to obtain predictive intervals.

Baselines: ARIMA (statsmodels) and optional Prophet baseline (if prophet is installed).

Evaluation: RMSE, MAE, Coverage (90% intervals), NRMSE, coverage-by-horizon, PIT histogram.

Exports: dataset CSV, model weights, TorchScript (if export succeeds), scaler params, diagnostic plots, comparison table, and a humanized run report.

Configuration (key parameters)

INPUT_LEN — input window length (default: 60 days)

OUTPUT_LEN — forecast horizon (default: 7 days)

MC_RUNS — Monte Carlo samples for uncertainty (default: 200)

EPOCHS, BATCH_SIZE, HIDDEN_SIZE, DROPOUT — training hyperparameters

OUTDIR — output directory for saved artifacts

Adjust these in the code before running for different experiments.

Output files (saved to OUTDIR)

sp500_close.csv — dataset used (downloaded or synthetic)

best_mc_lstm.pth — trained PyTorch weights

mc_lstm_traced.pt — TorchScript model (if export succeeded)

scaler_params.npz — saved scaler parameters

forecast_sample_comparison.jpg — sample observed vs predicted plot

coverage_by_horizon_lstm.jpg — coverage per horizon plot

pit_hist.jpg — PIT histogram

comparison_table.csv — metrics comparison (LSTM / ARIMA / Prophet)

run_report.txt — humanized project report

Notes & troubleshooting

yfinance download failures: In restricted environments, yfinance can fail. The script automatically uses a synthetic series so the pipeline runs end-to-end. For final evaluation, re-run in an environment with internet or provide a local CSV.

Prophet installation: prophet may require extra system dependencies and can take longer to install. If not available, the pipeline skips the Prophet baseline.

Torch version conflicts: Avoid force-reinstalling torch in Colab to prevent dependency issues with preinstalled packages (e.g., torchaudio).

ARIMA performance: Repeated ARIMA fits are slow. For stronger classical baselines use pmdarima.auto_arima offline.

Evaluation & interpretation

Point forecasts: RMSE and MAE across test windows.

Uncertainty: Coverage of 90% predictive intervals (fraction of true values inside intervals).

Calibration diagnostics: Coverage-by-horizon and PIT histogram. Use these to assess interval reliability across forecast lead times.

Recommendations for production

Use real S&P 500 data and re-run training/validation on the full series.

Persist scaler and model artifacts; export a serving-ready model (TorchScript or ONNX).

Serve via an API (FastAPI) exposing point forecasts and interval bounds.

Implement monitoring (prediction drift, interval calibration) and automated retraining.

Extending this work

Replace ARIMA(5,1,0) with auto_arima for order selection.

Add hyperparameter tuning (Optuna).

Train alternative architectures (TCN, Transformer).

Use ensembling or quantile regression for improved uncertainty estimates.

Add a backtesting framework and rolling evaluation.
