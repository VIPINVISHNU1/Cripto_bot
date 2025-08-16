import ccxt
import pandas as pd

exchange = ccxt.binance()  # or ccxt.bybit(), ccxt.kucoin(), etc.
bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=500)
df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Save to CSV
df.to_csv('BTCUSDT_4h_ohlcv.csv', index=False)

print("Saved 4h OHLCV data to BTCUSDT_4h_ohlcv.csv")
print(df)