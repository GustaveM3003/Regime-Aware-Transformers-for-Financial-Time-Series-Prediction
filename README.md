# Regime-Aware Transformers for Financial Time-Series Prediction

Virginia Tech — Deep Learning Project

> Gustave Miller · Naman Seth · Nithin Alluru

---

## Overview

Financial markets are inherently non-stationary, exhibiting regime shifts such as bull markets, bear markets, and periods of high volatility. This project investigates whether incorporating market regime information into a deep learning model improves return prediction and portfolio performance. We propose a regime-aware transformer that conditions its attention mechanism on the current market regime, predicting future asset returns from historical data and evaluating performance through risk-adjusted portfolio metrics.

---

## Repository Structure

```
├── data/
│   ├── raw/                  # Raw downloaded data
│   └── processed/            # Preprocessed features
├── models/
│   ├── transformer_a.py      # Baseline Transformer (A)
│   ├── transformer_b.py      # Transformer + Macro Concat (B)
│   └── transformer_c.py      # Transformer + Regime Mechanism (C)
├── portfolio/
│   └── strategy.py           # Portfolio construction and backtesting
├── experiments/
│   └── results/              # Saved metrics and outputs
├── README.md
└── requirements.txt
```

---

## Data Pipeline

> 🔧 **Owner: Naman Seth**

<!-- TODO: Document the full data pipeline here including:
  - Yahoo Finance data download (S&P 500 OHLCV, VIX, Treasury yields)
  - FRED-MD macroeconomic indicators
  - Handling of look-ahead bias
  - Train / Validation / Test split methodology and exact date ranges
  - Any survivorship bias considerations
-->

---

## Models

Three transformer variants are compared as a clean ablation. All models share the same preprocessing pipeline and are evaluated identically.

| Model | Description | Owner |
|-------|-------------|-------|
| A | Baseline Transformer — no regime information | Nithin |
| B | Transformer + Macro Concat — macro features appended at MLP head | Gustave |
| C | Transformer + Regime Mechanism — regime embedding injected into attention | Naman |

---

### Model A — Baseline Transformer

> **Owner: Nithin Alluru**

#### Preprocessing

**Step 1: Feature Engineering**

Convert all price-based features into log returns. Normalize volume as deviation from each stock's own rolling mean. Compute technical indicators on a rolling basis — never on the full dataset. Every feature must be constructed using only information available at time `t`. This is the look-ahead bias firewall.

```
Raw OHLCV        → Log Returns
Raw Volume       → Rolling Normalized Volume
Prices           → Technical Indicators (rolling only)

Output shape: [N stocks, T timesteps, F features]
```

**Step 2: Cross-Sectional Normalization**

At each timestep, z-score every feature across all N stocks. This ensures the model sees relative behavior only — how each stock compares to the universe on that day — never absolute levels.

```python
# For each timestep t, for each feature f:
mean = X[:, t, f].mean()
std  = X[:, t, f].std() + 1e-8
X[:, t, f] = (X[:, t, f] - mean) / std

# Output shape: [N stocks, T timesteps, F features]
```

#### Architecture

Each stock is processed independently through the following architecture:

```
Input:    [T, F]
          ↓
Linear Projection (F → d_model=64)
          ↓
+ Learnable Positional Embedding
          ↓
Transformer Encoder — 3 layers, 4 heads
  LayerNorm → Multi-Head Self-Attention → Residual
  LayerNorm → FFN (64 → 256 → 64) → Residual
          ↓
Mean Pool across T → [d_model]
          ↓
MLP Head: 64 → 32 → 1
          ↓
Output: predicted return scalar per stock
```

#### Hyperparameters

| Parameter           | Value         |
|---------------------|---------------|
| `d_model`           | 64            |
| Encoder layers      | 3             |
| Attention heads     | 4             |
| FFN dimension       | 256           |
| Dropout             | 0.1 – 0.2     |
| Positional encoding | Learnable     |
| Pooling             | Mean across T |
| Loss                | MSE           |

#### Training

| Setting           | Value            |
|-------------------|------------------|
| Loss              | MSE              |
| Optimizer         | AdamW            |
| Weight decay      | 1e-4 to 1e-3     |
| Gradient clipping | Clip norm 1.0    |
| Dropout           | 0.1 – 0.2        |
| Early stopping    | Monitor val loss |

#### Design Philosophy

Every decision traces back to one principle: **the model should learn relative behavior across stocks in return space, not absolute price levels.**

- **Log returns** enforce this on the price axis
- **Cross-sectional normalization** enforces this on the feature axis

The transformer learns the temporal patterns that make one stock's near-term return likely to be higher than another's. It never predicts whether the market goes up or down — only which stocks will outperform the rest.

---

### Model B — Transformer + Macro Concat

> 🔧 **Owner: Gustave Miller**

<!-- TODO: Document Model B here including:
  - Which macroeconomic features are appended and how
  - Where in the architecture the macro features are concatenated (e.g. at MLP head)
  - Any additional preprocessing specific to macro features
  - Hyperparameters that differ from Model A
-->

---

### Model C — Transformer + Regime Mechanism

> 🔧 **Owner: Naman Seth**

<!-- TODO: Document Model C here including:
  - How the regime representation is derived
  - How the regime embedding is injected into the attention mechanism
  - The regime embedding network architecture
  - Hyperparameters that differ from Model A
-->

---

## Portfolio Construction

<!-- TODO: Document the portfolio strategy here including:
  - Cross-sectional rank-based portfolio logic
  - How predicted returns are converted to long/short weights
  - Market-neutral constraint and verification
  - Position sizing methodology
-->

---

## Backtesting & Evaluation

<!-- TODO: Document the backtesting setup and results here including:
  - Test period and simulation methodology
  - Portfolio metrics: Sharpe ratio, maximum drawdown, annualized return, Sortino ratio, turnover
  - Predictive metrics: MSE, MAE, directional accuracy
  - Comparison against baselines: Linear Regression, LSTM, Buy-and-Hold, Equal Weight
  - Results table across all three models
-->

---
