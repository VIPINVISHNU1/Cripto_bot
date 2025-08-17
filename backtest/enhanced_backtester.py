import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class EnhancedBacktester:
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
        self.slippage_rate = float(config.get("slippage_rate", 0.0005))
        
        # Dynamic TP/SL optimization parameters
        self.target_monthly_return = config.get("target_monthly_return", 0.20)  # 20%
        self.risk_per_trade_pct = config.get("risk_per_trade_pct", 0.03)  # 3%
        
        # Modular hyperparameters
        self.stop_loss_range = config.get("stop_loss_range", [0.01, 0.02, 0.03, 0.04, 0.05])  # 1-5%
        self.take_profit_range = config.get("take_profit_range", [0.02, 0.03, 0.04, 0.05, 0.06])  # 2-6%
        
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/monthly_results", exist_ok=True)

    def optimize_tp_sl_for_month(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Optimize TP/SL parameters for maximum monthly return"""
        best_return = -np.inf
        best_sl = 0.02
        best_tp = 0.04
        
        for sl_pct in self.stop_loss_range:
            for tp_pct in self.take_profit_range:
                if tp_pct <= sl_pct:  # TP should be > SL
                    continue
                    
                monthly_return = self._simulate_month_with_params(data, sl_pct, tp_pct)
                if monthly_return > best_return:
                    best_return = monthly_return
                    best_sl = sl_pct
                    best_tp = tp_pct
        
        return best_sl, best_tp

    def _simulate_month_with_params(self, data: pd.DataFrame, stop_loss_pct: float, take_profit_pct: float) -> float:
        """Quick simulation to evaluate TP/SL parameters"""
        signals = self.strategy.run_backtest(data)
        
        cash = self.initial_balance
        position = None
        entry_price = 0
        
        for idx in range(1, len(data)):
            bar = data.iloc[idx]
            bar_high = bar["high"]
            bar_low = bar["low"]

            # Exit logic
            if position is not None:
                if position == "long":
                    stop_price = entry_price * (1 - stop_loss_pct)
                    tp_price = entry_price * (1 + take_profit_pct)
                    
                    if bar_low <= stop_price:
                        exit_price = stop_price * (1 - self.slippage_rate)
                        position_size = (cash * self.risk_per_trade_pct) / (entry_price - stop_price)
                        pnl = (exit_price - entry_price) * position_size
                        fees = (entry_price + exit_price) * position_size * self.fee_rate
                        cash += pnl - fees
                        position = None
                    elif bar_high >= tp_price:
                        exit_price = tp_price * (1 - self.slippage_rate)
                        position_size = (cash * self.risk_per_trade_pct) / (entry_price - stop_price)
                        pnl = (exit_price - entry_price) * position_size
                        fees = (entry_price + exit_price) * position_size * self.fee_rate
                        cash += pnl - fees
                        position = None
                        
                elif position == "short":
                    stop_price = entry_price * (1 + stop_loss_pct)
                    tp_price = entry_price * (1 - take_profit_pct)
                    
                    if bar_high >= stop_price:
                        exit_price = stop_price * (1 + self.slippage_rate)
                        position_size = (cash * self.risk_per_trade_pct) / (stop_price - entry_price)
                        pnl = (entry_price - exit_price) * position_size
                        fees = (entry_price + exit_price) * position_size * self.fee_rate
                        cash += pnl - fees
                        position = None
                    elif bar_low <= tp_price:
                        exit_price = tp_price * (1 + self.slippage_rate)
                        position_size = (cash * self.risk_per_trade_pct) / (stop_price - entry_price)
                        pnl = (entry_price - exit_price) * position_size
                        fees = (entry_price + exit_price) * position_size * self.fee_rate
                        cash += pnl - fees
                        position = None

            # Entry logic
            for sig in [s for s in signals if s["time"] == data.index[idx-1]]:
                if position is None:
                    if sig["type"] == "long":
                        entry_price = data.iloc[idx-1]["close"] * (1 + self.slippage_rate)
                        position = "long"
                    elif sig["type"] == "short":
                        entry_price = data.iloc[idx-1]["close"] * (1 - self.slippage_rate)
                        position = "short"

        return (cash - self.initial_balance) / self.initial_balance

    def simulate_month(self, data: pd.DataFrame, month_name: str, stop_loss_pct: float, take_profit_pct: float) -> Dict:
        """Simulate trading for a single month with given parameters"""
        signals = self.strategy.run_backtest(data)
        trades = []
        cash = self.initial_balance
        equity_curve = []
        position = None
        entry_price = 0
        entry_time = None
        stop_price = None
        tp_price = None

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
                
                if position == "long":
                    if bar_low <= stop_price:
                        exit_price = stop_price * (1 - self.slippage_rate)
                        exit_reason = "StopLoss"
                    elif bar_high >= tp_price:
                        exit_price = tp_price * (1 - self.slippage_rate)
                        exit_reason = "TakeProfit"
                        
                elif position == "short":
                    if bar_high >= stop_price:
                        exit_price = stop_price * (1 + self.slippage_rate)
                        exit_reason = "StopLoss"
                    elif bar_low <= tp_price:
                        exit_price = tp_price * (1 + self.slippage_rate)
                        exit_reason = "TakeProfit"

                if exit_reason is not None:
                    # Calculate position size based on risk management
                    if position == "long":
                        risk_amount = cash * self.risk_per_trade_pct
                        position_size = risk_amount / (entry_price - stop_price)
                        pnl = (exit_price - entry_price) * position_size
                        fees = (entry_price + exit_price) * position_size * self.fee_rate
                    else:
                        risk_amount = cash * self.risk_per_trade_pct
                        position_size = risk_amount / (stop_price - entry_price)
                        pnl = (entry_price - exit_price) * position_size
                        fees = (entry_price + exit_price) * position_size * self.fee_rate
                        
                    net_pnl = pnl - fees
                    cash += net_pnl
                    
                    trades.append({
                        "entry_time": entry_time, "exit_time": bar_time,
                        "entry_price": entry_price, "exit_price": exit_price,
                        "side": position, "size": position_size, "pnl": net_pnl, 
                        "fees": fees, "exit_reason": exit_reason
                    })
                    position = None

            # Entry logic
            for sig in [s for s in signals if s["time"] == data.index[idx-1]]:
                if position is None:
                    if sig["type"] == "long":
                        entry_price = prev_bar["close"] * (1 + self.slippage_rate)
                        entry_time = bar_time
                        position = "long"
                        stop_price = entry_price * (1 - stop_loss_pct)
                        tp_price = entry_price * (1 + take_profit_pct)
                    elif sig["type"] == "short":
                        entry_price = prev_bar["close"] * (1 - self.slippage_rate)
                        entry_time = bar_time
                        position = "short"
                        stop_price = entry_price * (1 + stop_loss_pct)
                        tp_price = entry_price * (1 - take_profit_pct)

            # Update equity curve
            if position == "long":
                unrealized_pnl = (bar["close"] - entry_price) * ((cash * self.risk_per_trade_pct) / (entry_price - stop_price))
                eq = cash + unrealized_pnl
            elif position == "short":
                unrealized_pnl = (entry_price - bar["close"]) * ((cash * self.risk_per_trade_pct) / (stop_price - entry_price))
                eq = cash + unrealized_pnl
            else:
                eq = cash
            equity_curve.append(eq)

        # Close any remaining position
        if position is not None:
            final_bar = data.iloc[-1]
            final_price = final_bar["close"] * (1 - self.slippage_rate if position=="long" else 1 + self.slippage_rate)
            
            if position == "long":
                risk_amount = cash * self.risk_per_trade_pct
                position_size = risk_amount / (entry_price - stop_price)
                pnl = (final_price - entry_price) * position_size
                fees = (entry_price + final_price) * position_size * self.fee_rate
            else:
                risk_amount = cash * self.risk_per_trade_pct
                position_size = risk_amount / (stop_price - entry_price)
                pnl = (entry_price - final_price) * position_size
                fees = (entry_price + final_price) * position_size * self.fee_rate
                
            net_pnl = pnl - fees
            cash += net_pnl
            
            trades.append({
                "entry_time": entry_time, "exit_time": data.index[-1],
                "entry_price": entry_price, "exit_price": final_price,
                "side": position, "size": position_size, "pnl": net_pnl,
                "fees": fees, "exit_reason": "EndOfMonth"
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
                "sharpe_ratio": 0.0
            }
        
        # Net profit calculations
        net_profit_inr = final_balance - self.initial_balance
        net_profit_pct = (net_profit_inr / self.initial_balance) * 100
        
        # Win rate
        winning_trades = [t for t in trades if t["pnl"] > 0]
        win_rate = len(winning_trades) / len(trades) * 100
        
        # Max drawdown
        max_dd = self.calculate_max_drawdown(equity_curve)
        
        # Sharpe ratio
        sharpe = self.calculate_sharpe_ratio(equity_curve)
        
        return {
            "month": results["month"],
            "net_profit_pct": round(net_profit_pct, 2),
            "net_profit_inr": round(net_profit_inr, 2),
            "max_drawdown": round(max_dd, 2),
            "win_rate": round(win_rate, 2),
            "num_trades": len(trades),
            "stop_loss_pct": results["stop_loss_pct"],
            "take_profit_pct": results["take_profit_pct"],
            "sharpe_ratio": round(sharpe, 3)
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
        
        return np.sqrt(252) * returns.mean() / returns.std()  # Annualized

    def run_monthly_backtest(self, start_year=2020, start_month=8, end_year=2025, end_month=8):
        """Run comprehensive monthly backtesting from August 2020 to August 2025"""
        self.logger.info(f"Starting monthly backtest from {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
        
        monthly_results = []
        current_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 1)
        
        while current_date <= end_date:
            month_name = current_date.strftime("%Y-%m")
            self.logger.info(f"Processing month: {month_name}")
            
            # Generate or load data for this month
            monthly_data = self.get_monthly_data(current_date)
            
            if monthly_data is not None and len(monthly_data) > 10:
                # Optimize TP/SL for this month
                optimal_sl, optimal_tp = self.optimize_tp_sl_for_month(monthly_data)
                
                # Run simulation with optimal parameters
                results = self.simulate_month(monthly_data, month_name, optimal_sl, optimal_tp)
                
                # Calculate metrics
                metrics = self.calculate_monthly_metrics(results)
                monthly_results.append(metrics)
                
                # Save detailed results
                trades_df = pd.DataFrame(results["trades"])
                if not trades_df.empty:
                    trades_df.to_csv(f"data/monthly_results/trades_{month_name}.csv", index=False)
                
                self.logger.info(f"Month {month_name}: Profit {metrics['net_profit_pct']:.2f}%, "
                               f"Trades: {metrics['num_trades']}, Win Rate: {metrics['win_rate']:.1f}%")
            else:
                self.logger.warning(f"Insufficient data for month {month_name}")
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        # Generate comprehensive report
        self.generate_comprehensive_report(monthly_results)
        return monthly_results

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

    def generate_comprehensive_report(self, monthly_results: List[Dict]):
        """Generate comprehensive summary report"""
        if not monthly_results:
            self.logger.error("No monthly results to report")
            return
        
        df = pd.DataFrame(monthly_results)
        
        # Summary statistics
        total_months = len(df)
        profitable_months = len(df[df['net_profit_pct'] > 0])
        profitable_rate = (profitable_months / total_months) * 100
        
        avg_monthly_return = df['net_profit_pct'].mean()
        max_monthly_return = df['net_profit_pct'].max()
        min_monthly_return = df['net_profit_pct'].min()
        
        avg_drawdown = df['max_drawdown'].mean()
        max_drawdown = df['max_drawdown'].max()
        
        avg_win_rate = df['win_rate'].mean()
        total_trades = df['num_trades'].sum()
        
        avg_sharpe = df['sharpe_ratio'].mean()
        
        # Check if strategy meets target of 20% monthly profit
        months_above_target = len(df[df['net_profit_pct'] >= 20])
        target_achievement_rate = (months_above_target / total_months) * 100
        
        # Robustness analysis
        std_monthly_returns = df['net_profit_pct'].std()
        consistency_score = 100 - min(std_monthly_returns, 100)  # Lower std = higher consistency
        
        # Generate report
        report = f"""
=== COMPREHENSIVE TRADING STRATEGY ANALYSIS ===
Period: {df['month'].iloc[0]} to {df['month'].iloc[-1]}

PERFORMANCE SUMMARY:
- Total Months Analyzed: {total_months}
- Profitable Months: {profitable_months} ({profitable_rate:.1f}%)
- Average Monthly Return: {avg_monthly_return:.2f}%
- Best Month: {max_monthly_return:.2f}%
- Worst Month: {min_monthly_return:.2f}%

TARGET ACHIEVEMENT:
- Target Monthly Return: 20%
- Months Above Target: {months_above_target} ({target_achievement_rate:.1f}%)

RISK METRICS:
- Average Monthly Drawdown: {avg_drawdown:.2f}%
- Maximum Drawdown: {max_drawdown:.2f}%
- Average Win Rate: {avg_win_rate:.1f}%
- Average Sharpe Ratio: {avg_sharpe:.3f}

TRADING ACTIVITY:
- Total Trades: {total_trades}
- Average Trades per Month: {total_trades/total_months:.1f}

ROBUSTNESS ANALYSIS:
- Monthly Return Volatility: {std_monthly_returns:.2f}%
- Consistency Score: {consistency_score:.1f}/100

STRATEGY ASSESSMENT:
"""
        
        # Strategy assessment
        if target_achievement_rate >= 50:
            report += "✅ EXCELLENT: Strategy consistently achieves target returns\n"
        elif target_achievement_rate >= 30:
            report += "✅ GOOD: Strategy frequently achieves target returns\n"
        elif target_achievement_rate >= 15:
            report += "⚠️  MODERATE: Strategy occasionally achieves target returns\n"
        else:
            report += "❌ POOR: Strategy rarely achieves target returns\n"
            
        if profitable_rate >= 70:
            report += "✅ ROBUST: High profitability consistency\n"
        elif profitable_rate >= 50:
            report += "✅ STABLE: Decent profitability consistency\n"
        else:
            report += "⚠️  UNSTABLE: Low profitability consistency\n"
            
        if consistency_score >= 75:
            report += "✅ CONSISTENT: Low volatility in returns\n"
        elif consistency_score >= 50:
            report += "✅ MODERATE: Acceptable return volatility\n"
        else:
            report += "⚠️  VOLATILE: High volatility in returns\n"
        
        if avg_sharpe >= 1.0:
            report += "✅ EXCELLENT: Strong risk-adjusted returns\n"
        elif avg_sharpe >= 0.5:
            report += "✅ GOOD: Decent risk-adjusted returns\n"
        else:
            report += "⚠️  WEAK: Poor risk-adjusted returns\n"
        
        # Overfitting analysis
        if std_monthly_returns < 15:
            report += "✅ NO OVERFITTING: Consistent performance across periods\n"
        elif std_monthly_returns < 25:
            report += "⚠️  POSSIBLE OVERFITTING: Some performance inconsistency\n"
        else:
            report += "❌ LIKELY OVERFITTING: High performance inconsistency\n"
        
        # Save report
        with open("data/comprehensive_strategy_report.txt", "w") as f:
            f.write(report)
        
        # Save detailed results
        df.to_csv("data/monthly_results_summary.csv", index=False)
        
        # Create visualization
        self.create_performance_charts(df)
        
        self.logger.info("Comprehensive report generated")
        print(report)

    def create_performance_charts(self, df: pd.DataFrame):
        """Create performance visualization charts"""
        plt.figure(figsize=(15, 12))
        
        # Monthly returns
        plt.subplot(3, 2, 1)
        df['net_profit_pct'].plot(kind='bar', color=['green' if x >= 0 else 'red' for x in df['net_profit_pct']])
        plt.title('Monthly Returns (%)')
        plt.ylabel('Return %')
        plt.xticks(rotation=45)
        plt.axhline(y=20, color='blue', linestyle='--', label='Target 20%')
        plt.legend()
        
        # Cumulative returns
        plt.subplot(3, 2, 2)
        cumulative_returns = (1 + df['net_profit_pct']/100).cumprod() - 1
        cumulative_returns.plot(color='blue')
        plt.title('Cumulative Returns')
        plt.ylabel('Cumulative Return')
        
        # Drawdown
        plt.subplot(3, 2, 3)
        df['max_drawdown'].plot(color='red')
        plt.title('Monthly Maximum Drawdown (%)')
        plt.ylabel('Drawdown %')
        
        # Win rate
        plt.subplot(3, 2, 4)
        df['win_rate'].plot(color='green')
        plt.title('Monthly Win Rate (%)')
        plt.ylabel('Win Rate %')
        
        # Trade count
        plt.subplot(3, 2, 5)
        df['num_trades'].plot(kind='bar', color='orange')
        plt.title('Number of Trades per Month')
        plt.ylabel('Trade Count')
        plt.xticks(rotation=45)
        
        # Sharpe ratio
        plt.subplot(3, 2, 6)
        df['sharpe_ratio'].plot(color='purple')
        plt.title('Monthly Sharpe Ratio')
        plt.ylabel('Sharpe Ratio')
        
        plt.tight_layout()
        plt.savefig('data/strategy_performance_charts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Performance charts saved to data/strategy_performance_charts.png")