import pandas as pd

# Load your csv data
df = pd.read_csv("BTCUSDT_4h_ohlcv.csv", parse_dates=["timestamp"])

fvg_list = []
trades = []

# Loop through candles to detect FVGs with "loose" logic (matches TradingView visual style)
for i in range(2, len(df)):
    c1 = df.iloc[i-2]
    c3 = df.iloc[i]

    # Bullish FVG: c1.low > c3.high (no/minimal threshold)
    if c1['low'] > c3['high']:
        fvg = {
            "type": "bullish",
            "upper": c3['high'],
            "lower": c1['low'],
            "created_idx": i,
            "touched": False
        }
        fvg_list.append(fvg)
    # Bearish FVG: c1.high < c3.low (no/minimal threshold)
    if c1['high'] < c3['low']:
        fvg = {
            "type": "bearish",
            "upper": c1['high'],
            "lower": c3['low'],
            "created_idx": i,
            "touched": False
        }
        fvg_list.append(fvg)

# For each FVG, look for first touch/entry after creation
for fvg in fvg_list:
    idx_after_fvg = fvg["created_idx"] + 1
    if idx_after_fvg >= len(df):
        continue
    for i in range(idx_after_fvg, len(df)):
        bar = df.iloc[i]
        if fvg["touched"]:
            break
        # Entry on first touch/entry (no strict close required)
        if fvg["type"] == "bullish" and bar["low"] <= fvg["upper"]:
            trades.append({
                "time": df.iloc[i]["timestamp"],
                "type": "long",
                "reason": "FVG_touch"
            })
            fvg["touched"] = True
        if fvg["type"] == "bearish" and bar["high"] >= fvg["lower"]:
            trades.append({
                "time": df.iloc[i]["timestamp"],
                "type": "short",
                "reason": "FVG_touch"
            })
            fvg["touched"] = True

print(f"Total FVGs detected: {len(fvg_list)}")
print(f"Total trades triggered: {len(trades)}")
print("Sample trades:")
print(pd.DataFrame(trades).head(10))