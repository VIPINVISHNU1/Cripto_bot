import pandas as pd

class SMCFVGStrategy:
    def __init__(self, config, broker):
        self.config = config
        self.broker = broker
        self.imbalance_threshold = float(config.get("imbalance_threshold", 0.0002))

    def run_backtest(self, data: pd.DataFrame):
        """
        FVG-based SMC: 
        - Detect FVG (imbalance) between 1st and 3rd candle of a rolling window.
        - Mark the FVG zone.
        - When price returns to the FVG, enter on first bounce (confirmation by close).
        """
        signals = []
        fvgs = []

        for i in range(2, len(data)):
            c1 = data.iloc[i-2]
            c2 = data.iloc[i-1]
            c3 = data.iloc[i]

            # Bullish FVG: c1.low > c3.high + threshold
            if c1['low'] > c3['high'] + (self.imbalance_threshold * c3['high']):
                fvg = {
                    "type": "bullish",
                    "upper": c3['high'],
                    "lower": c1['low'],
                    "created_idx": i,
                    "touched": False
                }
                fvgs.append(fvg)

            # Bearish FVG: c1.high < c3.low - threshold
            if c1['high'] < c3['low'] - (self.imbalance_threshold * c3['low']):
                fvg = {
                    "type": "bearish",
                    "upper": c1['high'],
                    "lower": c3['low'],
                    "created_idx": i,
                    "touched": False
                }
                fvgs.append(fvg)

        for fvg in fvgs:
            idx_after_fvg = fvg["created_idx"] + 1
            if idx_after_fvg >= len(data):
                continue
            for i in range(idx_after_fvg, len(data)):
                bar = data.iloc[i]
                if fvg["touched"]:
                    break

                # For bullish FVG: price low enters FVG zone and closes above upper (bounce)
                if fvg["type"] == "bullish" and bar["low"] <= fvg["upper"] and bar["high"] >= fvg["upper"]:
                    if bar["close"] > fvg["upper"]:
                        signals.append({
                            "time": data.index[i],
                            "type": "long",
                            "reason": "FVG_bounce"
                        })
                        fvg["touched"] = True

                # For bearish FVG: price high enters FVG zone and closes below lower (bounce)
                elif fvg["type"] == "bearish" and bar["high"] >= fvg["lower"] and bar["low"] <= fvg["lower"]:
                    if bar["close"] < fvg["lower"]:
                        signals.append({
                            "time": data.index[i],
                            "type": "short",
                            "reason": "FVG_bounce"
                        })
                        fvg["touched"] = True

        signals = sorted(signals, key=lambda x: x["time"])
        return signals