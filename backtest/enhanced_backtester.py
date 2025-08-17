import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class EnhancedBacktester:
    """Enhanced backtester with dynamic risk management and advanced metrics"""
    
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

        # Risk management settings
        self.account_risk_pct = float(config.get("account_risk_pct", 0.01))  # 1% account risk per trade
        self.max_position_size = float(config.get("max_position_size", 0.1))  # Max 10% of account
        
        # Fallback stop loss and take profit (if strategy doesn't provide dynamic levels)
        self.stop_loss_pct = float(config.get("stop_loss_pct", 0.01))
        self.take_profit_pct = float(config.get("take_profit_pct", 0.02))

        os.makedirs("data", exist_ok=True)

    def calculate_position_size(self, entry_price, stop_price, account_balance):
        """Calculate position size based on account risk percentage"""
        if stop_price <= 0 or entry_price <= 0:
            return self.position_size
            
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_price)
        
        # Calculate position size based on account risk
        risk_amount = account_balance * self.account_risk_pct
        position_size = risk_amount / risk_per_unit
        
        # Cap position size
        max_size = account_balance * self.max_position_size / entry_price
        position_size = min(position_size, max_size)
        
        # Ensure minimum size
        position_size = max(position_size, self.min_order_size)
        
        return position_size

    def run(self):
        self.logger.info(f"Starting enhanced backtest for {self.symbol} from {self.start} to {self.end} on {self.timeframe}")
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
        position_size = 0
        min_size = self.min_order_size

        for idx in range(1, len(data)):
            bar = data.iloc[idx]
            prev_bar = data.iloc[idx-1]
            bar_time = data.index[idx]
            bar_high = bar["high"]
            bar_low = bar["low"]

            # Manage open position: check stop loss / take profit
            if position is not None:
                exit_reason = None
                exit_price = None
                
                # For long position
                if position == "long":
                    # Check stop loss first
                    if bar_low <= stop_price:
                        exit_price = stop_price * (1 - self.slippage_rate)
                        exit_reason = "StopLoss"
                    # Check take profit
                    elif bar_high >= tp_price:
                        exit_price = tp_price * (1 - self.slippage_rate)
                        exit_reason = "TakeProfit"
                        
                # For short position
                elif position == "short":
                    # Check stop loss first
                    if bar_high >= stop_price:
                        exit_price = stop_price * (1 + self.slippage_rate)
                        exit_reason = "StopLoss"
                    # Check take profit
                    elif bar_low <= tp_price:
                        exit_price = tp_price * (1 + self.slippage_rate)
                        exit_reason = "TakeProfit"

                if exit_reason is not None:
                    if position == "long":
                        pnl = (exit_price - entry_price) * position_size
                        fees = (entry_price + exit_price) * position_size * self.fee_rate
                    else:
                        pnl = (entry_price - exit_price) * position_size
                        fees = (entry_price + exit_price) * position_size * self.fee_rate
                    
                    net_pnl = pnl - fees
                    cash += net_pnl
                    
                    trades.append({
                        "entry_time": entry_time, "exit_time": bar_time,
                        "entry_price": entry_price, "exit_price": exit_price,
                        "stop_price": stop_price, "tp_price": tp_price,
                        "side": position, "size": position_size, "pnl": net_pnl, "fees": fees,
                        "exit_reason": exit_reason
                    })
                    
                    position = None

            # Check for new signals
            for sig in signals:
                sig_time = sig["time"]
                if sig_time != bar_time:
                    continue
                    
                if position is None:  # Only enter if no position
                    # Get dynamic levels from signal if available
                    if "stop_price" in sig and "tp_price" in sig:
                        entry_price = sig.get("entry_price", prev_bar["close"])
                        stop_price = sig["stop_price"]
                        tp_price = sig["tp_price"]
                    else:
                        # Fallback to percentage-based levels
                        if sig["type"] == "long":
                            entry_price = prev_bar["close"] * (1 + self.slippage_rate)
                            stop_price = entry_price * (1 - self.stop_loss_pct)
                            tp_price = entry_price * (1 + self.take_profit_pct)
                        else:
                            entry_price = prev_bar["close"] * (1 - self.slippage_rate)
                            stop_price = entry_price * (1 + self.stop_loss_pct)
                            tp_price = entry_price * (1 - self.take_profit_pct)
                    
                    # Calculate dynamic position size
                    if hasattr(self.strategy, 'get_dynamic_levels'):
                        position_size = self.calculate_position_size(entry_price, stop_price, cash)
                    else:
                        position_size = max(np.floor(self.position_size / min_size) * min_size, min_size)
                    
                    entry_time = bar_time
                    position = sig["type"]

            # Update equity curve
            if position == "long":
                eq = cash + (bar["close"] - entry_price) * position_size
            elif position == "short":
                eq = cash + (entry_price - bar["close"]) * position_size
            else:
                eq = cash
            equity_curve.append(eq)

        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "dates": data.index[1:].tolist()
        }

    def report(self, results, tag="test"):
        trades = results["trades"]
        eq_curve = results["equity_curve"]
        dates = results["dates"]

        if not trades:
            self.logger.info("No trades executed.")
            return

        # Basic metrics
        pnl = [t["pnl"] for t in trades]
        total_pnl = sum(pnl)
        win_trades = [p for p in pnl if p > 0]
        loss_trades = [p for p in pnl if p <= 0]
        
        win_count = len(win_trades)
        loss_count = len(loss_trades)
        total_trades = len(pnl)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Advanced metrics
        avg_win = np.mean(win_trades) if win_trades else 0
        avg_loss = np.mean(loss_trades) if loss_trades else 0
        profit_factor = sum(win_trades) / abs(sum(loss_trades)) if loss_trades else float('inf')
        
        # Expectancy calculation
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Risk metrics
        returns = np.diff(eq_curve) / eq_curve[:-1]
        downside_returns = returns[returns < 0]
        
        # Sortino ratio (using downside deviation)
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino = np.sqrt(252) * np.mean(returns) / downside_std if downside_std > 0 else 0
        else:
            sortino = float('inf') if np.mean(returns) > 0 else 0
        
        max_dd, _ = self.max_drawdown(eq_curve)
        sharpe = self.sharpe_ratio(eq_curve)
        
        # Trade analysis
        winning_streaks = []
        losing_streaks = []
        current_streak = 0
        current_type = None
        
        for pnl_val in pnl:
            if pnl_val > 0:
                if current_type == "win":
                    current_streak += 1
                else:
                    if current_type == "loss" and current_streak > 0:
                        losing_streaks.append(current_streak)
                    current_streak = 1
                    current_type = "win"
            else:
                if current_type == "loss":
                    current_streak += 1
                else:
                    if current_type == "win" and current_streak > 0:
                        winning_streaks.append(current_streak)
                    current_streak = 1
                    current_type = "loss"
        
        # Add final streak
        if current_type == "win":
            winning_streaks.append(current_streak)
        elif current_type == "loss":
            losing_streaks.append(current_streak)
        
        max_winning_streak = max(winning_streaks) if winning_streaks else 0
        max_losing_streak = max(losing_streaks) if losing_streaks else 0
        
        # Report results
        self.logger.info(f"Total P&L: {total_pnl:.2f}, Trades: {total_trades}, Win Rate: {win_rate:.2%}")
        self.logger.info(f"Avg Win: {avg_win:.2f}, Avg Loss: {avg_loss:.2f}, Profit Factor: {profit_factor:.2f}")
        self.logger.info(f"Expectancy: {expectancy:.2f}")
        self.logger.info(f"Max Drawdown: {max_dd:.2f}, Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}")
        self.logger.info(f"Max Win Streak: {max_winning_streak}, Max Loss Streak: {max_losing_streak}")

        # Plot equity curve and drawdown
        plt.figure(figsize=(12,8))
        
        plt.subplot(2,1,1)
        plt.plot(dates, eq_curve, label="Equity Curve", linewidth=2)
        plt.axhline(y=self.initial_balance, color='gray', linestyle='--', alpha=0.7, label="Initial Balance")
        plt.title(f"Equity Curve ({tag}) - Total P&L: {total_pnl:.2f}")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2,1,2)
        drawdown = self.drawdown_series(eq_curve)
        plt.plot(dates, drawdown, color='red', label="Drawdown", linewidth=2)
        plt.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
        plt.title(f"Drawdown - Max: {max_dd:.2f}")
        plt.ylabel("Drawdown")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"data/enhanced_equity_{tag}.png", dpi=150, bbox_inches='tight')
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