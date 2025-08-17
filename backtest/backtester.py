import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class Backtester:
    def __init__(self, config, strategy, logger):
        self.config = config
        self.strategy = strategy
        self.logger = logger
        self.broker = strategy.broker
        self.symbol = config["strategy"]["symbol"]
        self.timeframe = config["strategy"]["timeframe"]
        self.start = config["backtest"]["start"]
        self.end = config["backtest"]["end"]
        self.position_size = float(config["strategy"]["position_size"])
        self.min_order_size = float(config.get("min_order_size", 0.001))
        self.fee_rate = float(config.get("fee_rate", 0.001))
        self.slippage_rate = float(config.get("slippage_rate", 0.0005))
        self.train_pct = float(config.get("in_sample_pct", 0.7))
        self.initial_balance = float(config.get("initial_balance", 10000))

        # Stop loss and take profit settings
        self.stop_loss_pct = float(config.get("stop_loss_pct", 0.01))  # 1% default
        self.take_profit_pct = float(config.get("take_profit_pct", 0.02))  # 2% default

        os.makedirs("data", exist_ok=True)

    def run(self):
        self.logger.info(f"Starting backtest for {self.symbol} from {self.start} to {self.end} on {self.timeframe}")
        data = self.broker.get_historical_klines(self.symbol, self.timeframe, self.start, self.end)
        if data is None or data.empty:
            self.logger.error("No data retrieved for backtest.")
            return

        split_idx = int(len(data) * self.train_pct)
        data_in_sample = data.iloc[:split_idx]
        data_out_sample = data.iloc[split_idx:]

        self.logger.info(f"Running in-sample backtest on {len(data_in_sample)} bars ({self.train_pct*100:.0f}%)")
        results_in = self.simulate_trades(data_in_sample, "in_sample")

        self.logger.info(f"Running out-of-sample backtest on {len(data_out_sample)} bars ({(1-self.train_pct)*100:.0f}%)")
        results_out = self.simulate_trades(data_out_sample, "out_sample")

        self.logger.info("In-sample results:")
        self.report(results_in, "in_sample")

        self.logger.info("Out-of-sample results:")
        self.report(results_out, "out_sample")

    def simulate_trades(self, data, tag="test"):
        signals = self.strategy.run_backtest(data)
        trades = []
        cash = self.initial_balance
        equity_curve = []
        position = None
        entry_price = 0
        entry_time = None
        stop_price = None
        tp_price = None
        min_size = self.min_order_size

        for idx in range(1, len(data)):
            bar = data.iloc[idx]
            prev_bar = data.iloc[idx-1]
            bar_time = data.index[idx]
            bar_high = bar["high"]
            bar_low = bar["low"]

            # Manage open position: check stop loss / take profit
            if position is not None:
                # Check SL/TP first
                exit_reason = None
                exit_price = None
                # For long
                if position == "long":
                    # Check stop loss
                    if bar_low <= stop_price:
                        exit_price = stop_price * (1 - self.slippage_rate)
                        exit_reason = "StopLoss"
                    # Check take profit
                    elif bar_high >= tp_price:
                        exit_price = tp_price * (1 - self.slippage_rate)
                        exit_reason = "TakeProfit"
                # For short
                elif position == "short":
                    # Check stop loss
                    if bar_high >= stop_price:
                        exit_price = stop_price * (1 + self.slippage_rate)
                        exit_reason = "StopLoss"
                    # Check take profit
                    elif bar_low <= tp_price:
                        exit_price = tp_price * (1 + self.slippage_rate)
                        exit_reason = "TakeProfit"

                if exit_reason is not None:
                    size = max(np.floor(self.position_size / min_size) * min_size, min_size)
                    if position == "long":
                        pnl = (exit_price - entry_price) * size
                        fees = (entry_price + exit_price) * size * self.fee_rate
                    else:
                        pnl = (entry_price - exit_price) * size
                        fees = (entry_price + exit_price) * size * self.fee_rate
                    net_pnl = pnl - fees
                    cash += net_pnl
                    trades.append({
                        "entry_time": entry_time, "exit_time": bar_time,
                        "entry_price": entry_price, "exit_price": exit_price,
                        "side": position, "size": size, "pnl": net_pnl, "fees": fees,
                        "exit_reason": exit_reason
                    })
                    position = None
                    entry_price = None
                    entry_time = None
                    stop_price = None
                    tp_price = None

            # Only use signals generated at previous bar close (no peeking)
            for sig in [s for s in signals if s["time"] == data.index[idx-1]]:
                if position is None:
                    size = max(np.floor(self.position_size / min_size) * min_size, min_size)
                    if sig["type"] == "long":
                        entry_price = prev_bar["close"] * (1 + self.slippage_rate)
                        entry_time = bar_time
                        position = "long"
                        stop_price = entry_price * (1 - self.stop_loss_pct)
                        tp_price = entry_price * (1 + self.take_profit_pct)
                    elif sig["type"] == "short":
                        entry_price = prev_bar["close"] * (1 - self.slippage_rate)
                        entry_time = bar_time
                        position = "short"
                        stop_price = entry_price * (1 + self.stop_loss_pct)
                        tp_price = entry_price * (1 - self.take_profit_pct)
                # If already in position, consider exit on opposite signal (market exit)
                elif (position == "long" and sig["type"] == "short") or (position == "short" and sig["type"] == "long"):
                    size = max(np.floor(self.position_size / min_size) * min_size, min_size)
                    exit_price = prev_bar["close"] * (1 - self.slippage_rate if position=="long" else 1 + self.slippage_rate)
                    if position == "long":
                        pnl = (exit_price - entry_price) * size
                        fees = (entry_price + exit_price) * size * self.fee_rate
                    else:
                        pnl = (entry_price - exit_price) * size
                        fees = (entry_price + exit_price) * size * self.fee_rate
                    net_pnl = pnl - fees
                    cash += net_pnl
                    trades.append({
                        "entry_time": entry_time, "exit_time": bar_time,
                        "entry_price": entry_price, "exit_price": exit_price,
                        "side": position, "size": size, "pnl": net_pnl, "fees": fees,
                        "exit_reason": "SignalReverse"
                    })
                    # Now reverse
                    if sig["type"] == "long":
                        entry_price = prev_bar["close"] * (1 + self.slippage_rate)
                        entry_time = bar_time
                        position = "long"
                        stop_price = entry_price * (1 - self.stop_loss_pct)
                        tp_price = entry_price * (1 + self.take_profit_pct)
                    elif sig["type"] == "short":
                        entry_price = prev_bar["close"] * (1 - self.slippage_rate)
                        entry_time = bar_time
                        position = "short"
                        stop_price = entry_price * (1 + self.stop_loss_pct)
                        tp_price = entry_price * (1 - self.take_profit_pct)

            # Update equity curve
            size = max(np.floor(self.position_size / min_size) * min_size, min_size)
            if position == "long":
                eq = cash + (bar["close"] - entry_price) * size
            elif position == "short":
                eq = cash + (entry_price - bar["close"]) * size
            else:
                eq = cash
            equity_curve.append(eq)

        # If in position at the end, close at last price
        if position is not None:
            final_bar = data.iloc[-1]
            final_price = final_bar["close"] * (1 - self.slippage_rate if position=="long" else 1 + self.slippage_rate)
            size = max(np.floor(self.position_size / min_size) * min_size, min_size)
            if position == "long":
                pnl = (final_price - entry_price) * size
                fees = (entry_price + final_price) * size * self.fee_rate
            else:
                pnl = (entry_price - final_price) * size
                fees = (entry_price + final_price) * size * self.fee_rate
            net_pnl = pnl - fees
            cash += net_pnl
            trades.append({
                "entry_time": entry_time, "exit_time": data.index[-1],
                "entry_price": entry_price, "exit_price": final_price,
                "side": position, "size": size, "pnl": net_pnl, "fees": fees,
                "exit_reason": "EndOfTest"
            })
            equity_curve.append(cash)

        dates = data.index[1:1+len(equity_curve)]
        if len(dates) < len(equity_curve):
            equity_curve = equity_curve[:len(dates)]
        elif len(dates) > len(equity_curve):
            dates = dates[:len(equity_curve)]

        tag = tag or "test"
        pd.DataFrame(trades).to_csv(f"data/trades_{tag}.csv", index=False)
        return {
            "trades": trades,
            "final_balance": cash,
            "equity_curve": equity_curve,
            "dates": dates
        }

    def report(self, results, tag="test"):
        trades = results["trades"]
        eq_curve = results["equity_curve"]
        dates = results["dates"]

        if not trades:
            self.logger.info("No trades executed.")
            return

        pnl = [t["pnl"] for t in trades]
        total = sum(pnl)
        win = len([p for p in pnl if p > 0])
        loss = len([p for p in pnl if p <= 0])
        win_rate = win / len(pnl) if pnl else 0
        max_dd, _ = self.max_drawdown(eq_curve)
        sharpe = self.sharpe_ratio(eq_curve)
        sortino = self.sortino_ratio(eq_curve)
        profit_factor = self.profit_factor(trades)
        expectancy = self.expectancy(trades)
        
        avg_win = np.mean([p for p in pnl if p > 0]) if win > 0 else 0
        avg_loss = np.mean([p for p in pnl if p <= 0]) if loss > 0 else 0
        
        self.logger.info(f"Total P&L: {total:.2f}, Trades: {len(trades)}, Win Rate: {win_rate:.2%}")
        self.logger.info(f"Avg Win: {avg_win:.2f}, Avg Loss: {avg_loss:.2f}")
        self.logger.info(f"Profit Factor: {profit_factor:.2f}, Expectancy: {expectancy:.2f}")
        self.logger.info(f"Max Drawdown: {max_dd:.2f}, Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}")

        # Plot equity curve and drawdown
        plt.figure(figsize=(12,6))
        plt.subplot(2,1,1)
        plt.plot(dates, eq_curve, label="Equity Curve")
        plt.title(f"Equity Curve ({tag})")
        plt.legend()
        plt.subplot(2,1,2)
        drawdown = self.drawdown_series(eq_curve)
        plt.plot(dates, drawdown, color='red', label="Drawdown")
        plt.title("Drawdown")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"data/equity_curve_{tag}.png")
        plt.close()

    @staticmethod
    def max_drawdown(equity):
        arr = np.array(equity)
        highs = np.maximum.accumulate(arr)
        dd = (arr - highs)
        min_dd = dd.min()
        min_idx = dd.argmin()
        return -min_dd, min_idx

    @staticmethod
    def drawdown_series(equity):
        arr = np.array(equity)
        highs = np.maximum.accumulate(arr)
        dd = (arr - highs)
        return dd

    @staticmethod
    def sharpe_ratio(equity, risk_free=0):
        returns = np.diff(equity) / equity[:-1]
        excess = returns - risk_free/252
        if returns.std() == 0:
            return 0
        return np.sqrt(252) * excess.mean() / returns.std()
    
    @staticmethod
    def sortino_ratio(equity, risk_free=0):
        """Calculate Sortino ratio using downside deviation"""
        returns = np.diff(equity) / equity[:-1]
        excess = returns - risk_free/252
        downside_returns = excess[excess < 0]
        if len(downside_returns) == 0:
            return float('inf') if excess.mean() > 0 else 0
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0
        return np.sqrt(252) * excess.mean() / downside_std
    
    @staticmethod
    def profit_factor(trades):
        """Calculate profit factor (gross profit / gross loss)"""
        pnl = [t["pnl"] for t in trades]
        wins = [p for p in pnl if p > 0]
        losses = [p for p in pnl if p <= 0]
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    @staticmethod
    def expectancy(trades):
        """Calculate expectancy per trade"""
        if not trades:
            return 0
        pnl = [t["pnl"] for t in trades]
        wins = [p for p in pnl if p > 0]
        losses = [p for p in pnl if p <= 0]
        
        win_rate = len(wins) / len(pnl)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        return (win_rate * avg_win) + ((1 - win_rate) * avg_loss)