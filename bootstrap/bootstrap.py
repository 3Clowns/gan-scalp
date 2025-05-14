import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# -----------------------------
# 1. Read & prepare data
# -----------------------------

data = pd.read_csv('./data/stocks_lkoh.csv')
data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
data['date'] = pd.to_datetime(data['date'])
ohlcv_cols = ['open', 'high', 'low', 'close']
data = data[['date'] + ohlcv_cols]
returns_data = pd.DataFrame({'date': data['date']})
for col in ohlcv_cols:
    returns_data[col] = data[col] / data[col].shift(1)
# -----------------------------
# 2. STL Decomposition
# -----------------------------

def bootstrap_residuals(blocks, n, block_length):

    n_blocks = int(np.ceil(n / block_length)) + 2

    synthetic_resid = []
    for _ in range(n_blocks):
        block = blocks[np.random.randint(0, len(blocks))]
        synthetic_resid.extend(block)
    synthetic_resid = np.array(synthetic_resid)[:n]
    return synthetic_resid

seasonal_period = 2600
block_length = 520
B = 10
synthetic_prices = {}
synthetic_returns = {}
# -----------------------------
# 3. Plotting trend, seasonality, residuals & distribution
# -----------------------------

for col in ohlcv_cols:
    print(f"Processing column: {col}")
    series = returns_data[col].dropna()
    n_obs = len(series)

    stl = STL(series, period=seasonal_period, robust=True)
    result = stl.fit()
    print('done')
    trend = result.trend.values
    seasonal = result.seasonal.values
    resid = result.resid.values

    blocks = [resid[i:i + block_length] for i in range(len(resid) - block_length + 1)]

    synthetic_prices[col] = []
    synthetic_returns[col] = []
    dates_returns = data['date'].iloc[1:].reset_index(drop=True)
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(dates_returns, trend, label='Trend', color='tab:blue')
    plt.title(f"{col.capitalize()} - Trend Component")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(dates_returns, seasonal, label='Seasonal', color='tab:orange')
    plt.title(f"{col.capitalize()} - Seasonal Component")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(dates_returns, resid, label='Residual', color='tab:green')
    plt.title(f"{col.capitalize()} - Residual Component")
    plt.legend()
    for b in range(B):
        synthetic_resid = bootstrap_residuals(blocks, n_obs, block_length)
        synthetic_ret = trend + seasonal + synthetic_resid
        synthetic_returns[col].append(synthetic_ret)
        initial_price = data[col].iloc[0]
        synthetic_price_series = initial_price * np.cumprod(synthetic_ret)
        synthetic_prices[col].append(synthetic_price_series)

    plt.figure(figsize=(12, 6))
    plt.plot(data['date'].iloc[1:], data[col].iloc[1:], label='Original')
    plt.plot(data['date'].iloc[1:], synthetic_prices[col][0], label='Synthetic', alpha=0.7)
    plt.title(f'Original vs Synthetic Series for {col}')
    plt.xlabel('Date')
    plt.ylabel(f'{col.capitalize()} Value')
    plt.legend()
    plt.show()

combined_df = pd.DataFrame()
# -----------------------------
# 4. Saving to csv
# -----------------------------
dates_returns = data['date'].iloc[1:].reset_index(drop=True)
rep_dfs = []
for rep in range(B):
    rep_dict = {col: synthetic_prices[col][rep] for col in ohlcv_cols}
    rep_dict['date'] = dates_returns
    rep_dict['rep']  = rep + 1
    df_rep = pd.DataFrame(rep_dict)
    rep_dfs.append(df_rep)

combined_df = pd.concat(rep_dfs, axis=0, ignore_index=True)
combined_df = combined_df[ohlcv_cols]

output_filename = 'bootstrapdata.csv'
combined_df.to_csv(output_filename, index=False)
print(f"Saved all synthetic results to {output_filename}")