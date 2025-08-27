# Mean Reversion Strategy in Crude Oil Futures

## Project Overview
This project implements and evaluates a **mean reversion trading strategy** on Crude Oil (WTI) futures using exponentially weighted moving averages, rolling volatilities, and z-score signals.  
The goal is to test whether crude oil exhibits mean-reverting behavior and to assess the strategy’s performance under **realistic execution assumptions** (transaction costs, leverage caps, and short-sale collateral).

---

## Methodology
1. **Data**  
   - Daily front-month WTI futures (`CL=F`) from Yahoo Finance over the sample period 2015–2025
   - Returns calculated and analyzed for distributional properties.

2. **Statistical Tests**  
   - ADF tests on price  to assess stationarity.  
   - Autocorrelation (ACF/PACF) and Jarque-Bera tests on returns.

3. **Signal Construction**  
   - Rolling EWMA mean (50-day, exponential weighting).  
   - Rolling volatility for standardization.  
   - Z-score deviation from mean as trading signal.  
   - Entry: long if `Z < -k`; short if `Z > k`.  
   - Exit: when Z-score reverts toward zero 

4. **Execution Model**  
   - Fixed notional risk (50% of equity).  
   - Transaction costs: 5 bps per trade.  
   - Short proceeds held as collateral (not reused).  
   - Leverage cap: 2x equity.  
   - Daily mark-to-market accounting.

---

## Results

- Sharpe Ratio: **0.34**  
- Max Drawdown: **-79.6%**  
- Total Trades: **31**  
- Turnover/day: **0.0205%**

---

# Conclusions
- Outright crude oil prices are **not stationary** (ADF p≈0.30).  
- Mean reversion in crude oil is fragile; performance deteriorates in high-volatility regimes (e.g. April 2020 negative WTI shock).  
- Robust execution modeling (collateral, leverage caps) is critical to avoid unrealistic equity curves.
- Strategy is still suspectible to large drawdowns.

---

## ⚡ Next Steps
- Extend strategy to **calendar spreads (CL1–CL2)** or **inter-commodity spreads (WTI–Brent)**.  
- Test Ornstein-Uhlenbeck (OU) process fitting instead of rolling z-scores.  
- Implement walk-forward optimization to reduce overfitting risk.  
- Add volatility and regime filters.
- Implement mechanism to reduce maximum drawdown
---


