import pandas as pd

class SMCImbalanceOrderBlockStrategy:
    def __init__(self, config, broker):
        self.config = config
        self.broker = broker

        # Sensible default; you can override via config
        self.imbalance_threshold = float(config.get("imbalance_threshold", 0.0002))  # 0.07% gap

    def run_backtest(self, data: pd.DataFrame):
        """
        Generate signals based on order block and imbalance logic:
        - For each rolling set of 3 candles, look for a significant gap between 1st and 3rd candle.
        - Mark the first candle as the order block.
        - When price returns to the order block after the imbalance, generate a trade signal.
        """
        signals = []
        order_blocks = []

        for i in range(2, len(data)):
            c1 = data.iloc[i-2]
            c2 = data.iloc[i-1]
            c3 = data.iloc[i]

            # Bullish imbalance: c1.low > c3.high by a threshold
            if c1['low'] > c3['high'] * (1 + self.imbalance_threshold):
                ob_price = c1['low']
                order_blocks.append({
                    "type": "bullish",
                    "level": ob_price,
                    "created_idx": i-2,
                    "imbalance_idx": i,
                    "touched": False
                })

            # Bearish imbalance: c1.high < c3.low by a threshold
            if c1['high'] < c3['low'] * (1 - self.imbalance_threshold):
                ob_price = c1['high']
                order_blocks.append({
                    "type": "bearish",
                    "level": ob_price,
                    "created_idx": i-2,
                    "imbalance_idx": i,
                    "touched": False
                })

        # Now scan forward: price returns to OB after imbalance, generate signal
        for ob in order_blocks:
            idx_after_imbalance = ob["imbalance_idx"] + 1
            if idx_after_imbalance >= len(data):
                continue
            for i in range(idx_after_imbalance, len(data)):
                bar = data.iloc[i]
                if ob["touched"]:
                    break
                # For bullish OB, look for price touching or dipping below the OB level
                if ob["type"] == "bullish" and bar["low"] <= ob["level"]:
                    signals.append({
                        "time": data.index[i],
                        "type": "long",
                        "reason": "OB_imbalance_touch"
                    })
                    ob["touched"] = True
                # For bearish OB, look for price touching or exceeding the OB level
                elif ob["type"] == "bearish" and bar["high"] >= ob["level"]:
                    signals.append({
                        "time": data.index[i],
                        "type": "short",
                        "reason": "OB_imbalance_touch"
                    })
                    ob["touched"] = True

        # Optional: sort signals by time just in case
        signals = sorted(signals, key=lambda x: x["time"])

        return signals