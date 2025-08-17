import pandas as pd

class SMCFVGLooseStrategy:
    def __init__(self, config, broker=None):
        self.config = config
        self.broker = broker

    def run_backtest(self, data: pd.DataFrame):
        fvg_list = []
        trades = []

        # Ensure data index is datetime for signal timing
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex (timestamp)!")

        # Loose FVG detection
        for i in range(2, len(data)):
            c1 = data.iloc[i-2]
            c3 = data.iloc[i]
            # Bullish FVG
            if c1['low'] > c3['high']:
                fvg = {
                    "type": "bullish",
                    "upper": c3['high'],
                    "lower": c1['low'],
                    "created_idx": i,
                    "touched": False
                }
                fvg_list.append(fvg)
            # Bearish FVG
            if c1['high'] < c3['low']:
                fvg = {
                    "type": "bearish",
                    "upper": c1['high'],
                    "lower": c3['low'],
                    "created_idx": i,
                    "touched": False
                }
                fvg_list.append(fvg)

        # Entry on first touch after FVG creation
        for fvg in fvg_list:
            idx_after_fvg = fvg["created_idx"] + 1
            if idx_after_fvg >= len(data):
                continue
            for i in range(idx_after_fvg, len(data)):
                bar = data.iloc[i]
                if fvg["touched"]:
                    break
                if fvg["type"] == "bullish" and bar["low"] <= fvg["upper"]:
                    trades.append({
                        "time": data.index[i],
                        "type": "long",
                        "reason": "FVG_touch",
                        "bar_index": i
                    })
                    fvg["touched"] = True
                if fvg["type"] == "bearish" and bar["high"] >= fvg["lower"]:
                    trades.append({
                        "time": data.index[i],
                        "type": "short",
                        "reason": "FVG_touch",
                        "bar_index": i
                    })
                    fvg["touched"] = True

        print(f"FVGs detected: {len(fvg_list)}, Trades generated: {len(trades)}")
        return trades