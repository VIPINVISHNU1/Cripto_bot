import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Any, Optional


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_trades(self, trades: List[Dict[str, Any]], initial_balance: float = 10000) -> Dict[str, Any]:
        """
        Analyze trading performance and generate comprehensive metrics.
        """
        if not trades:
            return {"error": "No trades to analyze"}
        
        df = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = df['pnl'].sum()
        total_fees = df['fees'].sum()
        net_pnl = total_pnl
        
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        largest_win = df['pnl'].max()
        largest_loss = df['pnl'].min()
        
        # Risk metrics
        profit_factor = abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] <= 0]['pnl'].sum()) if losing_trades > 0 else float('inf')
        
        # Calculate equity curve
        equity_curve = [initial_balance]
        for pnl in df['pnl']:
            equity_curve.append(equity_curve[-1] + pnl)
        
        # Drawdown analysis
        equity_array = np.array(equity_curve)
        peak_array = np.maximum.accumulate(equity_array)
        drawdown_array = (equity_array - peak_array) / peak_array * 100
        max_drawdown = abs(drawdown_array.min())
        
        # Sharpe ratio (simplified)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Trade distribution by exit reason
        exit_reasons = df['exit_reason'].value_counts().to_dict()
        
        # Signal analysis
        signal_types = df['side'].value_counts().to_dict()
        
        # Hold time analysis
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df['hold_time'] = df['exit_time'] - df['entry_time']
        avg_hold_time = df['hold_time'].mean()
        
        # Position sizing analysis
        avg_position_size = df['size'].mean()
        max_position_size = df['size'].max()
        min_position_size = df['size'].min()
        
        analysis = {
            "summary": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": round(win_rate, 2),
                "total_pnl": round(total_pnl, 2),
                "total_fees": round(total_fees, 2),
                "net_pnl": round(net_pnl, 2),
                "final_balance": round(initial_balance + net_pnl, 2),
                "return_pct": round((net_pnl / initial_balance) * 100, 2)
            },
            "trade_metrics": {
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "largest_win": round(largest_win, 2),
                "largest_loss": round(largest_loss, 2),
                "profit_factor": round(profit_factor, 2),
                "avg_hold_time": str(avg_hold_time)
            },
            "risk_metrics": {
                "max_drawdown": round(max_drawdown, 2),
                "sharpe_ratio": round(sharpe_ratio, 2),
                "avg_position_size": round(avg_position_size, 4),
                "max_position_size": round(max_position_size, 4),
                "min_position_size": round(min_position_size, 4)
            },
            "distribution": {
                "exit_reasons": exit_reasons,
                "signal_types": signal_types
            },
            "equity_curve": equity_curve,
            "drawdown_series": drawdown_array.tolist()
        }
        
        return analysis
    
    def generate_report(self, analysis: Dict[str, Any], strategy_name: str = "Strategy") -> str:
        """
        Generate a formatted performance report.
        """
        if "error" in analysis:
            return f"Performance Report - {strategy_name}\n{'='*50}\nError: {analysis['error']}"
        
        summary = analysis["summary"]
        trade_metrics = analysis["trade_metrics"]
        risk_metrics = analysis["risk_metrics"]
        distribution = analysis["distribution"]
        
        report = f"""
Performance Report - {strategy_name}
{'='*50}

SUMMARY:
  Total Trades: {summary['total_trades']}
  Winning Trades: {summary['winning_trades']} ({summary['win_rate']}%)
  Losing Trades: {summary['losing_trades']}
  
  Total P&L: ${summary['total_pnl']:,.2f}
  Total Fees: ${summary['total_fees']:,.2f}
  Net P&L: ${summary['net_pnl']:,.2f}
  Return: {summary['return_pct']}%
  Final Balance: ${summary['final_balance']:,.2f}

TRADE METRICS:
  Average Win: ${trade_metrics['avg_win']:,.2f}
  Average Loss: ${trade_metrics['avg_loss']:,.2f}
  Largest Win: ${trade_metrics['largest_win']:,.2f}
  Largest Loss: ${trade_metrics['largest_loss']:,.2f}
  Profit Factor: {trade_metrics['profit_factor']}
  Average Hold Time: {trade_metrics['avg_hold_time']}

RISK METRICS:
  Max Drawdown: {risk_metrics['max_drawdown']}%
  Sharpe Ratio: {risk_metrics['sharpe_ratio']}
  Avg Position Size: {risk_metrics['avg_position_size']}
  Position Size Range: {risk_metrics['min_position_size']} - {risk_metrics['max_position_size']}

TRADE DISTRIBUTION:
  Exit Reasons: {distribution['exit_reasons']}
  Signal Types: {distribution['signal_types']}
"""
        
        return report
    
    def plot_equity_curve(self, equity_curve: List[float], title: str = "Equity Curve", filename: str = None):
        """
        Plot the equity curve.
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve, linewidth=2, color='blue')
            plt.title(title)
            plt.xlabel('Trade Number')
            plt.ylabel('Equity ($)')
            plt.grid(True, alpha=0.3)
            
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                self.logger.info(f"Equity curve saved to {filename}")
            
            plt.close()  # Close to save memory
            
        except Exception as e:
            self.logger.error(f"Error plotting equity curve: {e}")
    
    def plot_drawdown(self, drawdown_series: List[float], title: str = "Drawdown", filename: str = None):
        """
        Plot the drawdown series.
        """
        try:
            plt.figure(figsize=(12, 4))
            plt.fill_between(range(len(drawdown_series)), drawdown_series, 0, 
                           color='red', alpha=0.3, label='Drawdown')
            plt.plot(drawdown_series, color='red', linewidth=1)
            plt.title(title)
            plt.xlabel('Time')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                self.logger.info(f"Drawdown plot saved to {filename}")
            
            plt.close()  # Close to save memory
            
        except Exception as e:
            self.logger.error(f"Error plotting drawdown: {e}")
    
    def compare_strategies(self, results_dict: Dict[str, Dict[str, Any]]) -> str:
        """
        Compare multiple strategy results.
        """
        if len(results_dict) < 2:
            return "Need at least 2 strategies to compare"
        
        comparison = "Strategy Comparison\n" + "="*50 + "\n"
        
        metrics = ['total_trades', 'win_rate', 'net_pnl', 'return_pct', 'max_drawdown', 'sharpe_ratio']
        
        for metric in metrics:
            comparison += f"\n{metric.replace('_', ' ').title()}:\n"
            for strategy, analysis in results_dict.items():
                if metric in ['max_drawdown', 'sharpe_ratio']:
                    value = analysis.get('risk_metrics', {}).get(metric, 'N/A')
                else:
                    value = analysis.get('summary', {}).get(metric, 'N/A')
                comparison += f"  {strategy}: {value}\n"
        
        return comparison