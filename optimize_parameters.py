#!/usr/bin/env python3
"""
Parameter optimization script for trading strategies.
Runs systematic grid search and generates comprehensive analysis.
"""

import yaml
import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from broker.binance import BinanceBroker
from strategy.smc_fvg_loose_strategy import SMCFVGLooseStrategy
from utils.logger import get_logger
from optimization.parameter_optimizer import ParameterOptimizer

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def create_summary_report(best_results: dict, logger):
    """Create a human-readable summary report of optimization results."""
    
    if not best_results:
        logger.warning("No results to summarize")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"data/optimization_results/summary_report_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PARAMETER OPTIMIZATION SUMMARY REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        categories = [
            ("Best Out-of-Sample Sharpe Ratio", "best_out_sharpe"),
            ("Best Out-of-Sample Total P&L", "best_out_pnl"),
            ("Best Out-of-Sample Profit Factor", "best_out_profit_factor"),
            ("Most Consistent Performance", "most_consistent"),
            ("Best Risk-Adjusted Performance", "best_risk_adjusted")
        ]
        
        for title, key in categories:
            if key not in best_results:
                continue
                
            result = best_results[key]
            f.write(f"\n{title.upper()}\n")
            f.write("-" * len(title) + "\n")
            
            # Parameters
            f.write(f"Stop Loss: {result.get('stop_loss_pct', 'N/A'):.3f} ({result.get('stop_loss_pct', 0)*100:.1f}%)\n")
            f.write(f"Take Profit: {result.get('take_profit_pct', 'N/A'):.3f} ({result.get('take_profit_pct', 0)*100:.1f}%)\n")
            f.write(f"Position Size: {result.get('position_size', 'N/A'):.4f}\n")
            f.write(f"Imbalance Threshold: {result.get('imbalance_threshold', 'N/A'):.4f}\n\n")
            
            # Out-of-sample performance
            f.write("OUT-OF-SAMPLE PERFORMANCE:\n")
            f.write(f"  Total P&L: ${result.get('out_total_pnl', 0):.2f}\n")
            f.write(f"  Sharpe Ratio: {result.get('out_sharpe', 0):.2f}\n")
            f.write(f"  Max Drawdown: ${result.get('out_max_dd', 0):.2f}\n")
            f.write(f"  Win Rate: {result.get('out_win_rate', 0)*100:.1f}%\n")
            f.write(f"  Number of Trades: {result.get('out_trades', 0)}\n")
            f.write(f"  Profit Factor: {result.get('out_profit_factor', 0):.2f}\n")
            f.write(f"  Volatility: {result.get('out_volatility', 0):.3f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n")
        f.write("1. Use the 'Most Consistent Performance' parameters for live trading\n")
        f.write("2. Monitor performance closely and re-optimize if market conditions change\n")
        f.write("3. Consider the risk-adjusted performance for conservative trading\n")
        f.write("4. Ensure sufficient out-of-sample trades (>10) before implementing\n\n")
    
    logger.info(f"Summary report saved to {report_path}")
    print(f"\nSummary report saved to: {report_path}")

def main():
    print("="*60)
    print("CRYPTO TRADING STRATEGY PARAMETER OPTIMIZATION")
    print("="*60)
    
    # Load configuration
    config = load_config()
    logger = get_logger(config["logging"])
    
    print(f"Strategy: {config['strategy'].get('name', 'SMC FVG Loose')}")
    print(f"Symbol: {config['strategy']['symbol']}")
    print(f"Timeframe: {config['strategy']['timeframe']}")
    print(f"Date Range: {config['backtest']['start']} to {config['backtest']['end']}")
    
    # Initialize components
    broker = BinanceBroker(config["broker"], logger)
    
    # Patch broker for timestamp index
    orig_get_historical_klines = broker.get_historical_klines
    def get_historical_klines_with_index(symbol, timeframe, start, end):
        df = orig_get_historical_klines(symbol, timeframe, start, end)
        if df is not None and "timestamp" in df.columns:
            df = df.set_index("timestamp")
        return df
    broker.get_historical_klines = get_historical_klines_with_index
    
    strategy = SMCFVGLooseStrategy(config["strategy"], broker)
    
    # Initialize optimizer
    optimizer = ParameterOptimizer(config, strategy, broker, logger)
    
    # Get parameter grid info
    param_grid = optimizer.define_parameter_grid()
    total_combinations = len(optimizer.generate_parameter_combinations(param_grid))
    
    print(f"\nParameter Grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print(f"\nTotal combinations to test: {total_combinations}")
    
    # Ask user for confirmation
    max_combinations = input(f"\nEnter max combinations to test (press Enter for all {total_combinations}): ").strip()
    if max_combinations:
        try:
            max_combinations = int(max_combinations)
        except ValueError:
            max_combinations = None
    else:
        max_combinations = None
    
    print(f"\nStarting parameter optimization...")
    print("This may take a while depending on the number of combinations...")
    
    try:
        # Run optimization
        results = optimizer.optimize_parameters(max_combinations=max_combinations)
        
        if not results:
            print("No valid results obtained. Check logs for errors.")
            return
        
        print(f"\nOptimization completed! {len(results)} combinations tested.")
        
        # Analyze results
        print("Analyzing results...")
        best_results = optimizer.analyze_results()
        
        if best_results:
            print("Analysis completed!")
            
            # Create summary report
            create_summary_report(best_results, logger)
            
            # Print quick summary to console
            print("\n" + "="*50)
            print("QUICK SUMMARY")
            print("="*50)
            
            if 'most_consistent' in best_results:
                consistent = best_results['most_consistent']
                print("RECOMMENDED PARAMETERS (Most Consistent):")
                print(f"  Stop Loss: {consistent.get('stop_loss_pct', 0)*100:.1f}%")
                print(f"  Take Profit: {consistent.get('take_profit_pct', 0)*100:.1f}%")
                print(f"  Position Size: {consistent.get('position_size', 0):.4f}")
                print(f"  Imbalance Threshold: {consistent.get('imbalance_threshold', 0):.4f}")
                print(f"  Out-of-Sample Sharpe: {consistent.get('out_sharpe', 0):.2f}")
                print(f"  Out-of-Sample P&L: ${consistent.get('out_total_pnl', 0):.2f}")
            
            print(f"\nDetailed results saved in: data/optimization_results/")
            
        else:
            print("Analysis failed. Check logs for details.")
    
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()