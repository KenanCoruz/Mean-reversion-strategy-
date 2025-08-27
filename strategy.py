import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy.stats import jarque_bera


# Data 
symbol = "CL=F"  # Crude Oil
data = yf.download(symbol, start="2019-01-01", end="2025-01-01")[['Close']]
data.dropna(inplace=True)

# Fix for multi-index columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]
data['Close'] = data['Close'].astype(float)

# Analysis 
print(data.describe())

plt.figure(figsize=(12, 6))
plt.plot(data['Close'])
plt.title("Price Over Sammple Period")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# Distribution of returns
data['Returns'] = data['Close'].pct_change().dropna()
sns.histplot(data['Returns'].dropna(), bins=100, kde=True)
plt.xlim(-0.5, 0.5)
plt.title("Distribution of Daily Returns")
plt.show()

# Autocorrelation plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(data['Returns'].dropna(), lags=30)
plt.show()
plot_pacf(data['Returns'].dropna(), lags=30)
plt.show()


def adf_test(series):
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")

print("ADF Test on Price:")
adf_test(data['Close'])


# Strategy 
ewma_window = 50
vol_window = 50
zscore_threshold = 2

# Indicators
data['EWMA'] = data['Close'].ewm(span=ewma_window, adjust=False).mean().shift(1).astype(float)
data['Volatility'] = data['Close'].rolling(window=vol_window).std().shift(1).astype(float)
data['ZScore'] = ((data['Close'] - data['EWMA']) / data['Volatility']).astype(float)


# Signals
data['Signal'] = 0
data.loc[data['ZScore'] > zscore_threshold, 'Signal'] = -1
data.loc[data['ZScore'] < -zscore_threshold, 'Signal'] = 1

# BACKTEST WITH TRAIN/TEST SPLIT
split_date = "2022-01-01"
train = data.loc[:split_date].copy()
test = data.loc[split_date:].copy()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backtest_safe(
    df,
    price_col="Close", signal_col="Signal",
    init_cap=1_000_000,
    fee_bps=5,                      # 5 bps per side
    target_notional_pct=0.50,       # target 50% of equity per position
    max_leverage=2.0,
    max_units=None,                 
    min_hold_days=1                 
):


    fee = fee_bps / 10_000.0
    px = df[price_col].astype(float).values
    sig = df[signal_col].astype(int).values
    idx = df.index

    cash = float(init_cap)
    pos  = 0                       # +units long, -units short
    collateral = 0.0               # short proceeds parked, not spendable

    equity = []
    cash_series = []
    pos_series = []
    trades = []
    last_trade_i = -10**9

    for i in range(len(df)):
        p = float(px[i])
        s = int(sig[i])
        eq = cash + pos * p + collateral   # current equity before action

        
        desired = s

        # cooldown to prevent immediate churn
        if i - last_trade_i < min_hold_days:
            desired = np.sign(pos) if pos != 0 else 0

        # position sizing 
        # target notional = % of equity
        target_notional = target_notional_pct * max(eq, 1.0)
        target_units = int(target_notional // p)  # integer units

        # cap by leverage: |pos_after| * p <= max_leverage * equity
        max_units_lever = int((max_leverage * max(eq,1.0)) // p)
        if max_units is not None:
            max_units_lever = min(max_units_lever, int(max_units))
        target_units = min(target_units, max_units_lever)

        # position close logic 
        # If we must be long (1) but are short (<0), cover all first
        if desired == 1 and pos < 0:
            gross = abs(pos) * p
            tcost = gross * fee
            cash -= (gross + tcost)          # pay to buy-to-cover
            cash += collateral               # release collateral
            trades.append((idx[i], "COVER", abs(pos), p, -tcost))
            pos = 0
            collateral = 0.0
            last_trade_i = i

        # If we must be short (-1) but are long (>0), sell out first
        if desired == -1 and pos > 0:
            gross = pos * p
            tcost = gross * fee
            cash += (gross - tcost)
            trades.append((idx[i], "SELL", pos, p, -tcost))
            pos = 0
            last_trade_i = i

        # open position logic 
        # Open / adjust toward target only if flat
        if desired == 1 and pos == 0 and target_units > 0:
            notional = target_units * p
            tcost = notional * fee
            
            if cash >= (notional + tcost):
                cash -= (notional + tcost)
                pos = target_units
                trades.append((idx[i], "BUY", target_units, p, -tcost))
                last_trade_i = i

        if desired == -1 and pos == 0 and target_units > 0:
            notional = target_units * p
            tcost = notional * fee
            
            if cash >= tcost:
                cash -= tcost          # pay fee from cash
                pos = -target_units
                collateral = notional  # short proceeds locked as collateral
                trades.append((idx[i], "SHORT", target_units, p, -tcost))
                last_trade_i = i

        # mark to market 
        eq = cash + pos * p + collateral
        equity.append(eq)
        cash_series.append(cash)
        pos_series.append(pos)

    equity = pd.Series(equity, index=idx, name="Equity")
    ret = equity.pct_change().fillna(0.0)

    out = pd.DataFrame({
        "Price": df[price_col],
        "Signal": df[signal_col],
        "Position": pd.Series(pos_series, index=idx),
        "Cash": pd.Series(cash_series, index=idx),
        "Equity": equity,
        "Return": ret
    })

    trades_df = pd.DataFrame(trades, columns=["Time", "Action", "Units", "Price", "Fee"])
    return out, trades_df

bt, trades = backtest_safe(
    data,
    init_cap=1000000,
    fee_bps=5,
    target_notional_pct=0.50,
    max_leverage=2.0,
    max_units=10000,
    min_hold_days=1
)

rf = 0.02/252
ex = bt["Return"] - rf
sharpe = np.sqrt(252) * ex.mean() / ex.std() if ex.std() > 0 else 0.0
cum = bt["Equity"]
max_dd = (cum / cum.cummax() - 1.0).min()
turnover = trades.shape[0] / len(bt)  # trades per day

print(f"Sharpe: {sharpe:.3f} | MaxDD: {max_dd:.2%} | Trades: {len(trades)} | Turnover/day: {turnover:.4f}")

# plots
plt.figure(figsize=(12,6))
plt.plot(bt.index, bt["Equity"], color="black", label="Equity")
plt.title("Equity Curve (safe backtest)")
plt.xlabel("Date"); plt.ylabel("USD"); plt.legend(); plt.tight_layout(); plt.show()

import seaborn as sns
plt.figure(figsize=(10,5))
sns.histplot(bt["Return"].clip(-0.05,0.05), bins=80, kde=True, edgecolor="black")
plt.title("Daily Returns Distribution (clipped ±5%)")
plt.xlabel("Daily Return"); plt.tight_layout(); plt.show()
