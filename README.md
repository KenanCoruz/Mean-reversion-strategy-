# Mean Reversion Strategy in Crude Oil Futures

## 📌 Project Overview
This project implements and evaluates a **mean reversion trading strategy** on Crude Oil (WTI) futures using moving averages, rolling volatilities, and z-score signals.  
The goal is to test whether crude oil exhibits mean-reverting behavior and to assess the strategy’s performance under **realistic execution assumptions** (transaction costs, leverage caps, and short-sale collateral).

---

## 🔍 Methodology
1. **Data Source**  
   - Daily front-month WTI futures (`CL=F`) from Yahoo Finance (2015–2025).
   - Log returns calculated and analyzed for distributional properties.

2. **Statistical Tests**  
   - ADF and KPSS tests on price vs. spread to assess stationarity.  
   - Autocorrelation (ACF/PACF) and Jarque-Bera tests on returns.

3. **Signal Construction**  
   - Rolling EWMA mean (20-day, exponential weighting).  
   - Rolling volatility for standardization.  
   - Z-score deviation from mean as trading signal.  
   - Entry: long if `Z < -k`; short if `Z > k`.  
   - Exit: when Z-score reverts toward zero (hysteresis filter).

4. **Execution Model**  
   - Fixed notional risk (50% of equity).  
   - Transaction costs: 5 bps per trade.  
   - Short proceeds held as collateral (not reused).  
   - Leverage cap: 2x equity.  
   - Daily mark-to-market accounting.

---

## 📈 Results

### In-Sample (2015–2021)
- Sharpe Ratio: **0.91**  
- Max Drawdown: **-52%**  
- Total Trades: **X**  
- Turnover/day: **Y%**

### Out-of-Sample (2022–2025)
- Sharpe Ratio: **0.95**  
- Max Drawdown: **-27%**  
- Total Trades: **Z**

---

## 📊 Key Plots
### 1. Equity Curve
![Equity Curve](plots/equity_curve.png)

### 2. Distribution of Daily Returns
![Return Distribution](plots/return_distribution.png)

### 3. Z-Score Signal vs. Price
![Signals](plots/zscore_signals.png)

---

# Conclusions
- Outright crude oil prices are **not stationary** (ADF p≈0.23).  
- Mean reversion in crude oil is fragile; performance deteriorates in high-volatility regimes (e.g. April 2020 negative WTI shock).  
- Strategy improved when applied to **stationary spreads** (e.g., WTI calendar spread or WTI–Brent spread).  
- Robust execution modeling (collateral, leverage caps) is critical to avoid unrealistic equity curves.

---

## ⚡ Next Steps
- Extend strategy to **calendar spreads (CL1–CL2)** or **inter-commodity spreads (WTI–Brent)**.  
- Test Ornstein-Uhlenbeck (OU) process fitting instead of rolling z-scores.  
- Implement walk-forward optimization to reduce overfitting risk.  
- Add volatility and regime filters.

---

## 🛠️ How to Run
Clone repo and install requirements:
```bash
git clone https://github.com/YOUR_USERNAME/mean-reversion-crude-oil.git
cd mean-reversion-crude-oil
pip install -r requirements.txt
