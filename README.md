# DL4AI Final Project — Time-Series Data and Applications to Stock Markets

**CS313 Deep Learning for Artificial Intelligence — Spring 2026**
**Student:** Nguyen Dang Khoa | **ID:** 230087

---

## Project Overview

This project applies deep learning to stock market data from two markets: the **Nasdaq** (Apple Inc., AAPL) and the **Vietnamese stock market** (Banking sector). The project covers price forecasting, trading signal identification, and portfolio construction using CNN-LSTM models.

## Repository Structure

```
DL4AI-230087-project/
│
├── 230087-project-notebook.ipynb   # Main notebook — all Tasks 1 to 4
└── README.md                       # This file
```

## Model Architecture

All tasks use a **CNN-LSTM hybrid** architecture:

```
Conv1D(64, kernel=3) → MaxPooling1D(2)
Conv1D(128, kernel=3) → MaxPooling1D(2)
LSTM(64)
Dropout(0.2)
Dense(64, relu)
Dense(n_outputs)       ← 1 for Tasks 1–2, K for multi-step, sigmoid for Task 3
```

- **Conv1D layers** extract short-term local patterns (e.g. 3–5 day price formations)
- **LSTM layer** captures longer sequential dependencies
- **Task 3** uses binary cross-entropy loss with a sigmoid output and inverse-frequency class weights to handle label imbalance

---

## Key Design Decisions

### Normalisation
A global `MinMaxScaler` fitted **on the training split only** is applied to each company. Per-window normalisation (as in the original demo code) was found to be broken for stocks with large historical price ranges, producing MAPE values exceeding 17,000%.

### Train / Validation / Test Split
All splits are strictly **chronological (80 / 10 / 10)**. Random shuffling is never applied, as it would leak future information into training.

### Cross-Validation
`TimeSeriesSplit` with 5 folds (expanding window) is used throughout. Standard k-fold CV is not appropriate for time-series data.

### Company / Sector Selection
- **Task 1:** AAPL only, last 10 years of data (2012–2022)
- **Tasks 2–4:** Vietnamese Banks sector, companies with ≥ 2,400 rows (~10 years)
- **Task 3:** VCB (Vietcombank) — most liquid and longest-running bank stock

### Trading Signal Labels (Task 3)
- **Buy:** max return within next 10 days ≥ +2%
- **Sell:** min return within next 10 days ≤ −2%
- Class imbalance handled via inverse-frequency class weights

### Portfolio Construction (Task 4)
- **Aggressive:** top 5 stocks by predicted forward return
- **Conservative:** top 5 stocks by predicted return / risk score
- Weights allocated using **inverse-volatility** method (weight ∝ 1/σ)

---

## Evaluation Metrics

| Task type | Metrics used |
|-----------|-------------|
| Regression (Tasks 1, 2, 4) | RMSE, MAE, MAPE |
| Classification (Task 3) | F1, ROC-AUC, Precision, Recall, Confusion Matrix |
| Portfolio (Task 4) | Expected return, Risk score, Final score (return/risk) |
