import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class RobustBacktester:
    """
    Enhanced backtester with walk-forward optimization and robustness improvements
    """
    def __init__(self, config, strategy, logger):
        self.config = config
        self.strategy = strategy
        self.logger = logger
        self.broker = strategy.broker
        self.symbol = config["strategy"]["symbol"]
        self.timeframe = config["strategy"]["timeframe"]
        self.initial_balance = float(config.get("initial_balance", 10000))
        self.min_order_size = float(config.get("min_order_size", 0.001))
        self.fee_rate = float(config.get("fee_rate", 0.001))
        self.slippage_rate = float(config.get("slippage_rate", 0.001))
        
        # Enhanced parameters
        self.target_monthly_return = config.get("target_monthly_return", 0.15)  # 15%
        self.risk_per_trade_pct = config.get("risk_per_trade_pct", 0.02)  # 2%
        self.walk_forward = config.get("walk_forward_optimization", True)
        self.lookback_months = config.get("lookback_months", 6)
        
        # More conservative parameter ranges
        self.stop_loss_range = config.get("stop_loss_range", [0.015, 0.02, 0.025, 0.03])
        self.take_profit_range = config.get("take_profit_range", [0.02, 0.025, 0.03, 0.035])
        
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/robust_results", exist_ok=True)

    def walk_forward_optimize(self, historical_data: Dict[str, pd.DataFrame], test_month: str) -> Tuple[float, float]:
        """
        Walk-forward optimization: use last N months to optimize parameters for current month
        """
        test_date = datetime.strptime(test_month, "%Y-%m")
        
        # Get lookback months for optimization
        optimization_data = []
        for i in range(self.lookback_months):
            lookback_date = test_date.replace(day=1) - timedelta(days=32*i)
            lookback_month = lookback_date.strftime("%Y-%m")
            
            if lookback_month in historical_data:
                optimization_data.append(historical_data[lookback_month])
        
        if not optimization_data:
            # Fallback to default parameters
            return 0.02, 0.03
        
        # Combine historical data
        combined_data = pd.concat(optimization_data)
        
        # Optimize on combined historical data
        best_return = -np.inf
        best_sl = 0.02
        best_tp = 0.03
        
        for sl_pct in self.stop_loss_range:
            for tp_pct in self.take_profit_range:
                if tp_pct <= sl_pct * 1.2:  # Ensure minimum risk/reward ratio
                    continue
                    
                avg_return = self._evaluate_params_on_data(combined_data, sl_pct, tp_pct)
                if avg_return > best_return:
                    best_return = avg_return
                    best_sl = sl_pct
                    best_tp = tp_pct
        
        return best_sl, best_tp

    def _evaluate_params_on_data(self, data: pd.DataFrame, stop_loss_pct: float, take_profit_pct: float) -> float:
        """Evaluate parameters on historical data with robustness checks"""
        try:
            signals = self.strategy.run_backtest(data)
            
            if not signals:
                return -1.0
            
            cash = self.initial_balance
            trades = []
            position = None
            entry_price = 0
            
            for idx in range(1, len(data)):
                bar = data.iloc[idx]
                bar_high = bar["high"]
                bar_low = bar["low"]

                # Exit logic with volatility adjustment
                if position is not None:
                    # Adjust TP/SL based on recent volatility
                    recent_volatility = data['close'].pct_change().rolling(24).std().iloc[idx]
                    vol_adjustment = min(1 + recent_volatility, 1.5)  # Cap adjustment
                    
                    if position == "long":
                        adjusted_sl = entry_price * (1 - stop_loss_pct * vol_adjustment)
                        adjusted_tp = entry_price * (1 + take_profit_pct * vol_adjustment)
                        
                        if bar_low <= adjusted_sl:
                            exit_price = adjusted_sl * (1 - self.slippage_rate)
                            position_size = (cash * self.risk_per_trade_pct) / (entry_price - adjusted_sl)
                            pnl = (exit_price - entry_price) * position_size
                            fees = (entry_price + exit_price) * position_size * self.fee_rate
                            cash += pnl - fees
                            trades.append(pnl - fees)
                            position = None
                        elif bar_high >= adjusted_tp:
                            exit_price = adjusted_tp * (1 - self.slippage_rate)
                            position_size = (cash * self.risk_per_trade_pct) / (entry_price - adjusted_sl)
                            pnl = (exit_price - entry_price) * position_size
                            fees = (entry_price + exit_price) * position_size * self.fee_rate
                            cash += pnl - fees
                            trades.append(pnl - fees)
                            position = None
                            
                    elif position == "short":
                        adjusted_sl = entry_price * (1 + stop_loss_pct * vol_adjustment)
                        adjusted_tp = entry_price * (1 - take_profit_pct * vol_adjustment)
                        
                        if bar_high >= adjusted_sl:
                            exit_price = adjusted_sl * (1 + self.slippage_rate)
                            position_size = (cash * self.risk_per_trade_pct) / (adjusted_sl - entry_price)
                            pnl = (entry_price - exit_price) * position_size
                            fees = (entry_price + exit_price) * position_size * self.fee_rate
                            cash += pnl - fees
                            trades.append(pnl - fees)
                            position = None
                        elif bar_low <= adjusted_tp:
                            exit_price = adjusted_tp * (1 + self.slippage_rate)
                            position_size = (cash * self.risk_per_trade_pct) / (adjusted_sl - entry_price)
                            pnl = (entry_price - exit_price) * position_size
                            fees = (entry_price + exit_price) * position_size * self.fee_rate
                            cash += pnl - fees
                            trades.append(pnl - fees)
                            position = None

                # Entry logic with filters
                for sig in [s for s in signals if s["time"] == data.index[idx-1]]:
                    if position is None and len(trades) < 50:  # Limit trades to prevent overtrading
                        if sig["type"] == "long":
                            entry_price = data.iloc[idx-1]["close"] * (1 + self.slippage_rate)
                            position = "long"
                        elif sig["type"] == "short":
                            entry_price = data.iloc[idx-1]["close"] * (1 - self.slippage_rate)
                            position = "short"

            # Calculate metrics
            if not trades:
                return -1.0
            
            total_return = (cash - self.initial_balance) / self.initial_balance
            win_rate = len([t for t in trades if t > 0]) / len(trades)
            
            # Penalize strategies with too few trades or poor win rates
            if len(trades) < 5 or win_rate < 0.2:
                return total_return * 0.5
            
            return total_return
            
        except Exception as e:
            self.logger.warning(f"Error evaluating parameters: {e}")
            return -1.0

    def simulate_month_robust(self, data: pd.DataFrame, month_name: str, stop_loss_pct: float, take_profit_pct: float) -> Dict:
        """Enhanced month simulation with robustness improvements"""
        signals = self.strategy.run_backtest(data)
        trades = []
        cash = self.initial_balance
        equity_curve = []
        position = None
        entry_price = 0
        entry_time = None
        
        max_trades_per_month = 30  # Limit to prevent overtrading
        trades_this_month = 0

        for idx in range(1, len(data)):
            bar = data.iloc[idx]
            prev_bar = data.iloc[idx-1]
            bar_time = data.index[idx]
            bar_high = bar["high"]
            bar_low = bar["low"]

            # Calculate recent volatility for dynamic adjustments
            if idx >= 24:  # Need at least 24 periods for volatility calculation
                recent_volatility = data['close'].pct_change().rolling(24).std().iloc[idx]
                vol_adjustment = min(1 + recent_volatility * 2, 2.0)  # Cap at 2x
            else:
                vol_adjustment = 1.0

            # Manage open position with volatility-adjusted levels
            if position is not None:
                exit_reason = None
                exit_price = None
                
                if position == "long":
                    adjusted_sl = entry_price * (1 - stop_loss_pct * vol_adjustment)
                    adjusted_tp = entry_price * (1 + take_profit_pct * vol_adjustment)
                    
                    if bar_low <= adjusted_sl:
                        exit_price = adjusted_sl * (1 - self.slippage_rate)
                        exit_reason = "StopLoss"
                    elif bar_high >= adjusted_tp:
                        exit_price = adjusted_tp * (1 - self.slippage_rate)
                        exit_reason = "TakeProfit"
                        
                elif position == "short":
                    adjusted_sl = entry_price * (1 + stop_loss_pct * vol_adjustment)
                    adjusted_tp = entry_price * (1 - take_profit_pct * vol_adjustment)
                    
                    if bar_high >= adjusted_sl:
                        exit_price = adjusted_sl * (1 + self.slippage_rate)
                        exit_reason = "StopLoss"
                    elif bar_low <= adjusted_tp:
                        exit_price = adjusted_tp * (1 + self.slippage_rate)
                        exit_reason = "TakeProfit"

                if exit_reason is not None:
                    # Calculate position size with enhanced risk management
                    if position == "long":
                        risk_amount = cash * self.risk_per_trade_pct
                        price_diff = entry_price - adjusted_sl
                        position_size = risk_amount / price_diff if price_diff > 0 else 0
                        pnl = (exit_price - entry_price) * position_size
                        fees = (entry_price + exit_price) * position_size * self.fee_rate
                    else:
                        risk_amount = cash * self.risk_per_trade_pct
                        price_diff = adjusted_sl - entry_price
                        position_size = risk_amount / price_diff if price_diff > 0 else 0
                        pnl = (entry_price - exit_price) * position_size
                        fees = (entry_price + exit_price) * position_size * self.fee_rate
                        
                    net_pnl = pnl - fees
                    cash += net_pnl
                    
                    trades.append({
                        "entry_time": entry_time, "exit_time": bar_time,
                        "entry_price": entry_price, "exit_price": exit_price,
                        "side": position, "size": position_size, "pnl": net_pnl, 
                        "fees": fees, "exit_reason": exit_reason,
                        "vol_adjustment": vol_adjustment
                    })
                    position = None

            # Entry logic with additional filters
            for sig in [s for s in signals if s["time"] == data.index[idx-1]]:
                if (position is None and 
                    trades_this_month < max_trades_per_month and 
                    cash > self.initial_balance * 0.5):  # Don't trade if account is down too much
                    
                    if sig["type"] == "long":
                        entry_price = prev_bar["close"] * (1 + self.slippage_rate)
                        entry_time = bar_time
                        position = "long"
                        trades_this_month += 1
                    elif sig["type"] == "short":
                        entry_price = prev_bar["close"] * (1 - self.slippage_rate)
                        entry_time = bar_time
                        position = "short"
                        trades_this_month += 1

            # Update equity curve
            if position == "long":
                unrealized_pnl = (bar["close"] - entry_price) * ((cash * self.risk_per_trade_pct) / max(entry_price * stop_loss_pct, 0.001))
                eq = cash + unrealized_pnl
            elif position == "short":
                unrealized_pnl = (entry_price - bar["close"]) * ((cash * self.risk_per_trade_pct) / max(entry_price * stop_loss_pct, 0.001))
                eq = cash + unrealized_pnl
            else:
                eq = cash
            equity_curve.append(eq)

        # Close any remaining position
        if position is not None:
            final_bar = data.iloc[-1]
            final_price = final_bar["close"] * (1 - self.slippage_rate if position=="long" else 1 + self.slippage_rate)
            
            if position == "long":
                adjusted_sl = entry_price * (1 - stop_loss_pct)
                risk_amount = cash * self.risk_per_trade_pct
                position_size = risk_amount / (entry_price - adjusted_sl)
                pnl = (final_price - entry_price) * position_size
                fees = (entry_price + final_price) * position_size * self.fee_rate
            else:
                adjusted_sl = entry_price * (1 + stop_loss_pct)
                risk_amount = cash * self.risk_per_trade_pct
                position_size = risk_amount / (adjusted_sl - entry_price)
                pnl = (entry_price - final_price) * position_size
                fees = (entry_price + final_price) * position_size * self.fee_rate
                
            net_pnl = pnl - fees
            cash += net_pnl
            
            trades.append({
                "entry_time": entry_time, "exit_time": data.index[-1],
                "entry_price": entry_price, "exit_price": final_price,
                "side": position, "size": position_size, "pnl": net_pnl,
                "fees": fees, "exit_reason": "EndOfMonth",
                "vol_adjustment": 1.0
            })

        return {
            "month": month_name,
            "trades": trades,
            "final_balance": cash,
            "equity_curve": equity_curve,
            "dates": data.index[1:1+len(equity_curve)],
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct
        }

    def run_robust_backtest(self, start_year=None, start_month=None, end_year=None, end_month=None):
        """Run robust walk-forward backtesting"""
        # Use config values if provided, otherwise defaults
        start_year = start_year or self.config.get("test_start_year", 2020)
        start_month = start_month or self.config.get("test_start_month", 8)
        end_year = end_year or self.config.get("test_end_year", 2025)
        end_month = end_month or self.config.get("test_end_month", 8)
        
        self.logger.info(f"Starting robust walk-forward backtest from {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
        
        # First, collect all monthly data
        historical_data = {}
        current_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 1)
        
        self.logger.info("Collecting historical data...")
        while current_date <= end_date:
            month_name = current_date.strftime("%Y-%m")
            monthly_data = self.get_monthly_data(current_date)
            
            if monthly_data is not None and len(monthly_data) > 10:
                historical_data[month_name] = monthly_data
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        # Now run walk-forward optimization
        monthly_results = []
        current_date = datetime(start_year, start_month, 1)
        
        while current_date <= end_date:
            month_name = current_date.strftime("%Y-%m")
            
            if month_name in historical_data:
                self.logger.info(f"Processing month: {month_name}")
                
                # Use walk-forward optimization if enabled
                if self.walk_forward:
                    optimal_sl, optimal_tp = self.walk_forward_optimize(historical_data, month_name)
                else:
                    # Simple optimization on current month
                    optimal_sl, optimal_tp = self.optimize_tp_sl_for_month(historical_data[month_name])
                
                # Run simulation with optimal parameters
                results = self.simulate_month_robust(
                    historical_data[month_name], 
                    month_name, 
                    optimal_sl, 
                    optimal_tp
                )
                
                # Calculate metrics
                metrics = self.calculate_monthly_metrics(results)
                monthly_results.append(metrics)
                
                # Save detailed results
                trades_df = pd.DataFrame(results["trades"])
                if not trades_df.empty:
                    trades_df.to_csv(f"data/robust_results/trades_{month_name}.csv", index=False)
                
                self.logger.info(f"Month {month_name}: Profit {metrics['net_profit_pct']:.2f}%, "
                               f"Trades: {metrics['num_trades']}, Win Rate: {metrics['win_rate']:.1f}%, "
                               f"SL: {optimal_sl:.3f}, TP: {optimal_tp:.3f}")
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        # Generate enhanced report
        self.generate_robust_report(monthly_results)
        return monthly_results

    def optimize_tp_sl_for_month(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Conservative optimization for TP/SL parameters"""
        best_return = -np.inf
        best_sl = 0.02
        best_tp = 0.03
        
        for sl_pct in self.stop_loss_range:
            for tp_pct in self.take_profit_range:
                if tp_pct <= sl_pct * 1.2:  # Ensure minimum risk/reward
                    continue
                    
                monthly_return = self._evaluate_params_on_data(data, sl_pct, tp_pct)
                if monthly_return > best_return:
                    best_return = monthly_return
                    best_sl = sl_pct
                    best_tp = tp_pct
        
        return best_sl, best_tp

    def get_monthly_data(self, month_date: datetime) -> pd.DataFrame:
        """Get historical data for a specific month using the broker"""
        try:
            # Calculate month start and end
            start_date = month_date.replace(day=1)
            if month_date.month == 12:
                end_date = start_date.replace(year=start_date.year + 1, month=1)
            else:
                end_date = start_date.replace(month=start_date.month + 1)
            
            # Get data from broker
            data = self.broker.get_historical_klines(
                self.symbol, 
                self.timeframe, 
                start_date.strftime("%Y-%m-%d"), 
                end_date.strftime("%Y-%m-%d")
            )
            
            if data is not None and len(data) > 10:
                return data
            else:
                self.logger.warning(f"Insufficient data for {month_date.strftime('%Y-%m')}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting data for {month_date.strftime('%Y-%m')}: {e}")
            return None

    def calculate_monthly_metrics(self, results: Dict) -> Dict:
        """Calculate comprehensive metrics for a month"""
        trades = results["trades"]
        final_balance = results["final_balance"]
        equity_curve = results["equity_curve"]
        
        if not trades:
            return {
                "month": results["month"],
                "net_profit_pct": 0.0,
                "net_profit_inr": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "num_trades": 0,
                "stop_loss_pct": results["stop_loss_pct"],
                "take_profit_pct": results["take_profit_pct"],
                "sharpe_ratio": 0.0,
                "avg_vol_adjustment": 1.0,
                "profit_factor": 0.0
            }
        
        # Net profit calculations
        net_profit_inr = final_balance - self.initial_balance
        net_profit_pct = (net_profit_inr / self.initial_balance) * 100
        
        # Win rate and profit factor
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]
        win_rate = len(winning_trades) / len(trades) * 100
        
        gross_profit = sum(t["pnl"] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t["pnl"] for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Max drawdown
        max_dd = self.calculate_max_drawdown(equity_curve)
        
        # Sharpe ratio
        sharpe = self.calculate_sharpe_ratio(equity_curve)
        
        # Average volatility adjustment
        avg_vol_adj = np.mean([t.get("vol_adjustment", 1.0) for t in trades])
        
        return {
            "month": results["month"],
            "net_profit_pct": round(net_profit_pct, 2),
            "net_profit_inr": round(net_profit_inr, 2),
            "max_drawdown": round(max_dd, 2),
            "win_rate": round(win_rate, 2),
            "num_trades": len(trades),
            "stop_loss_pct": results["stop_loss_pct"],
            "take_profit_pct": results["take_profit_pct"],
            "sharpe_ratio": round(sharpe, 3),
            "avg_vol_adjustment": round(avg_vol_adj, 3),
            "profit_factor": round(profit_factor, 3)
        }

    def calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown percentage"""
        if not equity_curve:
            return 0.0
        
        arr = np.array(equity_curve)
        highs = np.maximum.accumulate(arr)
        drawdowns = (arr - highs) / highs * 100
        return abs(drawdowns.min())

    def calculate_sharpe_ratio(self, equity_curve):
        """Calculate Sharpe ratio"""
        if len(equity_curve) < 2:
            return 0.0
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * returns.mean() / returns.std()

    def generate_robust_report(self, monthly_results: List[Dict]):
        """Generate enhanced robustness report"""
        if not monthly_results:
            self.logger.error("No monthly results to report")
            return
        
        df = pd.DataFrame(monthly_results)
        
        # Enhanced statistics
        total_months = len(df)
        profitable_months = len(df[df['net_profit_pct'] > 0])
        profitable_rate = (profitable_months / total_months) * 100
        
        avg_monthly_return = df['net_profit_pct'].mean()
        median_monthly_return = df['net_profit_pct'].median()
        max_monthly_return = df['net_profit_pct'].max()
        min_monthly_return = df['net_profit_pct'].min()
        
        avg_drawdown = df['max_drawdown'].mean()
        max_drawdown = df['max_drawdown'].max()
        
        avg_win_rate = df['win_rate'].mean()
        total_trades = df['num_trades'].sum()
        
        avg_sharpe = df['sharpe_ratio'].mean()
        avg_profit_factor = df['profit_factor'].mean()
        
        # Target achievement analysis
        target_rate = self.target_monthly_return * 100
        months_above_target = len(df[df['net_profit_pct'] >= target_rate])
        target_achievement_rate = (months_above_target / total_months) * 100
        
        # Robustness metrics
        std_monthly_returns = df['net_profit_pct'].std()
        consistency_score = max(0, 100 - min(std_monthly_returns, 100))
        
        # Walk-forward analysis
        if len(df) >= 12:
            first_year = df.head(12)['net_profit_pct'].mean()
            last_year = df.tail(12)['net_profit_pct'].mean()
            performance_degradation = (first_year - last_year) / abs(first_year) * 100 if first_year != 0 else 0
        else:
            performance_degradation = 0
        
        # Generate comprehensive report
        report = f"""
=== ROBUST TRADING STRATEGY ANALYSIS ===
Period: {df['month'].iloc[0]} to {df['month'].iloc[-1]}
Walk-Forward Optimization: {'ENABLED' if self.walk_forward else 'DISABLED'}

PERFORMANCE SUMMARY:
- Total Months Analyzed: {total_months}
- Profitable Months: {profitable_months} ({profitable_rate:.1f}%)
- Average Monthly Return: {avg_monthly_return:.2f}%
- Median Monthly Return: {median_monthly_return:.2f}%
- Best Month: {max_monthly_return:.2f}%
- Worst Month: {min_monthly_return:.2f}%

TARGET ACHIEVEMENT:
- Target Monthly Return: {target_rate:.0f}%
- Months Above Target: {months_above_target} ({target_achievement_rate:.1f}%)

RISK METRICS:
- Average Monthly Drawdown: {avg_drawdown:.2f}%
- Maximum Drawdown: {max_drawdown:.2f}%
- Average Win Rate: {avg_win_rate:.1f}%
- Average Sharpe Ratio: {avg_sharpe:.3f}
- Average Profit Factor: {avg_profit_factor:.3f}

TRADING ACTIVITY:
- Total Trades: {total_trades}
- Average Trades per Month: {total_trades/total_months:.1f}

ROBUSTNESS ANALYSIS:
- Monthly Return Volatility: {std_monthly_returns:.2f}%
- Consistency Score: {consistency_score:.1f}/100
- Performance Degradation: {performance_degradation:.1f}%

STRATEGY ASSESSMENT:
"""
        
        # Enhanced assessment criteria
        if target_achievement_rate >= 60:
            report += "✅ EXCELLENT: Strategy consistently achieves target returns\n"
        elif target_achievement_rate >= 40:
            report += "✅ GOOD: Strategy frequently achieves target returns\n"
        elif target_achievement_rate >= 25:
            report += "⚠️  MODERATE: Strategy occasionally achieves target returns\n"
        else:
            report += "❌ POOR: Strategy rarely achieves target returns\n"
            
        if profitable_rate >= 75:
            report += "✅ HIGHLY ROBUST: Excellent profitability consistency\n"
        elif profitable_rate >= 60:
            report += "✅ ROBUST: Good profitability consistency\n"
        elif profitable_rate >= 50:
            report += "⚠️  MODERATE: Acceptable profitability consistency\n"
        else:
            report += "❌ UNSTABLE: Poor profitability consistency\n"
            
        if consistency_score >= 80:
            report += "✅ HIGHLY CONSISTENT: Very low volatility in returns\n"
        elif consistency_score >= 65:
            report += "✅ CONSISTENT: Acceptable return volatility\n"
        elif consistency_score >= 50:
            report += "⚠️  MODERATE: Some return volatility\n"
        else:
            report += "❌ VOLATILE: High volatility in returns\n"
        
        if avg_sharpe >= 1.5:
            report += "✅ EXCELLENT: Outstanding risk-adjusted returns\n"
        elif avg_sharpe >= 1.0:
            report += "✅ VERY GOOD: Strong risk-adjusted returns\n"
        elif avg_sharpe >= 0.5:
            report += "✅ GOOD: Decent risk-adjusted returns\n"
        else:
            report += "⚠️  WEAK: Poor risk-adjusted returns\n"
        
        # Overfitting analysis
        if abs(performance_degradation) < 10 and consistency_score > 70:
            report += "✅ ROBUST: No evidence of overfitting\n"
        elif abs(performance_degradation) < 20 and consistency_score > 50:
            report += "⚠️  MODERATE: Some evidence of overfitting\n"
        else:
            report += "❌ OVERFITTING: Strong evidence of parameter overfitting\n"
        
        if avg_profit_factor >= 1.5:
            report += "✅ EXCELLENT: Strong profit factor\n"
        elif avg_profit_factor >= 1.2:
            report += "✅ GOOD: Decent profit factor\n"
        else:
            report += "⚠️  WEAK: Low profit factor\n"
        
        # Save reports
        with open("data/robust_strategy_report.txt", "w") as f:
            f.write(report)
        
        df.to_csv("data/robust_monthly_results.csv", index=False)
        
        # Create enhanced visualizations
        self.create_robust_charts(df)
        
        self.logger.info("Robust analysis report generated")
        print(report)

    def create_robust_charts(self, df: pd.DataFrame):
        """Create enhanced performance visualization charts"""
        plt.figure(figsize=(16, 14))
        
        # Monthly returns with target line
        plt.subplot(4, 2, 1)
        colors = ['green' if x >= self.target_monthly_return*100 else 'orange' if x >= 0 else 'red' for x in df['net_profit_pct']]
        df['net_profit_pct'].plot(kind='bar', color=colors, alpha=0.7)
        plt.title('Monthly Returns (%) - Green: Above Target, Orange: Positive, Red: Negative')
        plt.ylabel('Return %')
        plt.xticks(rotation=45)
        plt.axhline(y=self.target_monthly_return*100, color='blue', linestyle='--', label=f'Target {self.target_monthly_return*100:.0f}%')
        plt.legend()
        
        # Cumulative returns
        plt.subplot(4, 2, 2)
        cumulative_returns = (1 + df['net_profit_pct']/100).cumprod() - 1
        cumulative_returns.plot(color='blue', linewidth=2)
        plt.title('Cumulative Returns')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        
        # Rolling win rate
        plt.subplot(4, 2, 3)
        df['win_rate'].rolling(6).mean().plot(color='green', label='6-Month Rolling Avg')
        df['win_rate'].plot(color='lightgreen', alpha=0.5, label='Monthly')
        plt.title('Win Rate Trend (%)')
        plt.ylabel('Win Rate %')
        plt.legend()
        plt.grid(True)
        
        # Drawdown analysis
        plt.subplot(4, 2, 4)
        df['max_drawdown'].plot(color='red', label='Monthly DD')
        df['max_drawdown'].rolling(3).mean().plot(color='maroon', label='3-Month Avg')
        plt.title('Drawdown Analysis (%)')
        plt.ylabel('Drawdown %')
        plt.legend()
        plt.grid(True)
        
        # Trade frequency
        plt.subplot(4, 2, 5)
        df['num_trades'].plot(kind='bar', color='orange', alpha=0.7)
        plt.title('Number of Trades per Month')
        plt.ylabel('Trade Count')
        plt.xticks(rotation=45)
        
        # Sharpe ratio trend
        plt.subplot(4, 2, 6)
        df['sharpe_ratio'].plot(color='purple', label='Monthly Sharpe')
        df['sharpe_ratio'].rolling(6).mean().plot(color='indigo', label='6-Month Avg')
        plt.title('Sharpe Ratio Trend')
        plt.ylabel('Sharpe Ratio')
        plt.legend()
        plt.grid(True)
        
        # Parameter evolution (if available)
        plt.subplot(4, 2, 7)
        df['stop_loss_pct'].plot(color='red', label='Stop Loss %', marker='o', markersize=3)
        df['take_profit_pct'].plot(color='green', label='Take Profit %', marker='s', markersize=3)
        plt.title('Dynamic Parameter Evolution')
        plt.ylabel('Parameter Value')
        plt.legend()
        plt.grid(True)
        
        # Profit factor trend
        plt.subplot(4, 2, 8)
        df['profit_factor'].plot(color='blue', label='Monthly PF')
        df['profit_factor'].rolling(3).mean().plot(color='navy', label='3-Month Avg')
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Break-even')
        plt.title('Profit Factor Trend')
        plt.ylabel('Profit Factor')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('data/robust_strategy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Enhanced performance charts saved to data/robust_strategy_analysis.png")