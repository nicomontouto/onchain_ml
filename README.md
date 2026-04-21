# Onchain ML — Uniswap Trading Strategy with ML

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-189fdd)

## Overview

End-to-end machine learning pipeline for building and backtesting algorithmic trading strategies on Uniswap (UNI/USDT). Combines **on-chain protocol metrics** (from The Graph and Dune Analytics) with **price data** (Binance) and **sentiment indicators** (Fear & Greed Index) to train XGBoost and LSTM classifiers that predict directional price moves.

The pipeline produces LONG/SHORT/FLAT signals and simulates a full capital account, benchmarked against Buy & Hold.

## Pipeline Architecture

```
fetch_data.py          → Download OHLCV (Binance), on-chain metrics (The Graph / Dune)
feature_engineering.py → Build technical + on-chain features
process_data.py        → Clean, merge, label (target: next-period return sign)
model_baseline.py      → XGBoost walk-forward validation + hyperparameter tuning
model_lstm.py          → LSTM sequence classifier
simulate_v3.py         → LONG/SHORT/FLAT simulation with real USD capital
generate_report_v3.py  → PDF report with equity curves, drawdowns, metrics
main_v3.py             → Orchestrator (runs full pipeline end-to-end)
```

## Strategy Variants (V3)

| Variant | Model | Signal logic |
|---------|-------|-------------|
| A | XGBoost | Fixed probability threshold |
| B | XGBoost | Dynamic threshold adjusted by Fear & Greed Index |
| C | LSTM | Fixed asymmetric threshold (long bias) |
| BH | — | Buy & Hold benchmark |

Frequencies: **4h** and **daily** (1h discarded after ablation).

## Features

**Price-based:** returns, log-returns, RSI, MACD, Bollinger Bands, ATR, volume ratios.

**On-chain (Uniswap v3):** TVL, swap volume, fee revenue, active positions, liquidity depth, holder count.

**Sentiment:** Fear & Greed Index (daily, forward-filled to intraday frequencies).

## Results

See `reporte_uni_ml_v3.pdf` for the full results report including:
- Walk-forward validation accuracy per frequency and model
- Equity curves vs Buy & Hold
- Drawdown profiles
- Feature importance rankings

Selected results from `data/figures/` and `data/figures_v3/` are also committed.

## Project Structure

```
onchain_ml/
├── fetch_data.py           # Data ingestion (Binance, The Graph, Dune, Fear & Greed)
├── feature_engineering.py  # Feature construction
├── process_data.py         # Data cleaning and labelling
├── model_baseline.py       # XGBoost walk-forward + tuning
├── model_lstm.py           # LSTM training
├── evaluate.py             # Evaluation metrics
├── simulate_v3.py          # V3 LONG/SHORT/FLAT backtester
├── generate_report_v3.py   # PDF report generation
├── main.py                 # V2 orchestrator
├── main_v3.py              # V3 orchestrator (recommended entry point)
├── best_lstm.pt            # Trained LSTM weights
├── reporte_uni_ml_v3.pdf   # Full results report (V3)
├── data/
│   ├── thegraph_uni_protocol.csv   # On-chain data snapshot
│   ├── dune_uni_holders.csv        # Holder count snapshot
│   ├── figures/                    # V2 result plots
│   └── figures_v3/                 # V3 result plots
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

# Run full pipeline
python main_v3.py

# Skip XGBoost retraining (use cached probabilities)
python main_v3.py --skip-xgb-retrain

# Force full retraining
python main_v3.py --no-cache
```

Data cache and processed files are generated locally in `data/cache/` and `data/processed/` (gitignored).

## Tech Stack

| Tool | Role |
|------|------|
| Python 3.10+ | Core language |
| XGBoost | Gradient boosted classifier |
| PyTorch | LSTM sequence model |
| scikit-learn | Walk-forward splitting, metrics |
| pandas / NumPy | Data wrangling |
| requests / pyarrow | API calls and Parquet I/O |
| fpdf2 | PDF report generation |
| matplotlib | Visualisations |

## Future Work

- Integrate live Binance WebSocket feed for paper trading.
- Add position sizing via Kelly criterion.
- Extend to other Uniswap pairs (ETH/USDC, WBTC/ETH).
- Reinforcement learning agent for dynamic threshold adaptation.
