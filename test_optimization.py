#!/usr/bin/env python3
"""
Test parameter optimization with local data (demo mode).
Uses local CSV data instead of connecting to Binance API.
"""

import yaml
import sys
import os
import pandas as pd
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy.smc_fvg_loose_strategy import SMCFVGLooseStrategy
from utils.logger import get_logger
from optimization.parameter_optimizer import ParameterOptimizer

class MockBroker:
    """Mock broker that uses local CSV data for testing."""
    
    def __init__(self, config, logger):
        self.logger = logger
        self.data_file = "BTCUSDT_4h_ohlcv.csv"
        
    def get_historical_klines(self, symbol, timeframe, start, end):
        """Load data from local CSV file."""
        try:
            df = pd.read_csv(self.data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by date range if specified
            if start:
                start_date = pd.to_datetime(start)
                df = df[df['timestamp'] >= start_date]
            if end:
                end_date = pd.to_datetime(end)
                df = df[df['timestamp'] <= end_date]
            
            # Return in the expected format
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype({
                'open': float, 'high': float, 'low': float, 'close': float, 'volume': float
            })
            
        except Exception as e:
            self.logger.error(f"Error loading data from {self.data_file}: {e}")
            return None

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    print("="*60)
    print("PARAMETER OPTIMIZATION TEST (DEMO MODE)")
    print("="*60)
    print("Using local CSV data for demonstration...")
    
    # Load configuration
    config = load_config()
    logger = get_logger(config["logging"])
    
    # Use smaller parameter grid for testing
    config["backtest"]["start"] = "2025-05-25"  # Use available data range
    config["backtest"]["end"] = "2025-08-16"
    
    print(f"Strategy: SMC FVG Loose")
    print(f"Symbol: {config['strategy']['symbol']}")
    print(f"Timeframe: {config['strategy']['timeframe']}")
    print(f"Date Range: {config['backtest']['start']} to {config['backtest']['end']}")
    
    # Initialize components with mock broker
    broker = MockBroker(config["broker"], logger)
    
    # Patch broker for timestamp index
    orig_get_historical_klines = broker.get_historical_klines
    def get_historical_klines_with_index(symbol, timeframe, start, end):
        df = orig_get_historical_klines(symbol, timeframe, start, end)
        if df is not None and "timestamp" in df.columns:
            df = df.set_index("timestamp")
        return df
    broker.get_historical_klines = get_historical_klines_with_index
    
    strategy = SMCFVGLooseStrategy(config["strategy"], broker)
    
    # Initialize optimizer with reduced parameter grid for demo
    optimizer = ParameterOptimizer(config, strategy, broker, logger)
    
    # Override parameter grid for faster testing
    def define_test_parameter_grid():
        return {
            'stop_loss_pct': [0.01, 0.015, 0.02],  # 3 values
            'take_profit_pct': [0.02, 0.025, 0.03],  # 3 values  
            'position_size': [0.001, 0.002],  # 2 values
            'imbalance_threshold': [0.0002, 0.0005]  # 2 values
        }
    
    optimizer.define_parameter_grid = define_test_parameter_grid
    
    # Get parameter grid info
    param_grid = optimizer.define_parameter_grid()
    combinations = optimizer.generate_parameter_combinations(param_grid)
    
    print(f"\nReduced Parameter Grid for Demo:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print(f"\nTotal combinations to test: {len(combinations)}")
    
    print(f"\nStarting parameter optimization...")
    
    try:
        # Run optimization
        results = optimizer.optimize_parameters(max_combinations=20)  # Limit for demo
        
        if not results:
            print("No valid results obtained. Check logs for errors.")
            return
        
        print(f"\nOptimization completed! {len(results)} combinations tested.")
        
        # Analyze results
        print("Analyzing results...")
        best_results = optimizer.analyze_results()
        
        if best_results:
            print("Analysis completed!")
            
            # Print summary to console
            print("\n" + "="*50)
            print("OPTIMIZATION RESULTS SUMMARY")
            print("="*50)
            
            categories = [
                ("Best Sharpe Ratio", "best_out_sharpe"),
                ("Best Total P&L", "best_out_pnl"),
                ("Most Consistent", "most_consistent")
            ]
            
            for title, key in categories:
                if key in best_results:
                    result = best_results[key]
                    print(f"\n{title}:")
                    print(f"  Stop Loss: {result.get('stop_loss_pct', 0)*100:.1f}%")
                    print(f"  Take Profit: {result.get('take_profit_pct', 0)*100:.1f}%")
                    print(f"  Position Size: {result.get('position_size', 0):.4f}")
                    print(f"  Imbalance Threshold: {result.get('imbalance_threshold', 0):.4f}")
                    print(f"  Out-Sample Sharpe: {result.get('out_sharpe', 0):.2f}")
                    print(f"  Out-Sample P&L: ${result.get('out_total_pnl', 0):.2f}")
                    print(f"  Out-Sample Trades: {result.get('out_trades', 0)}")
            
            print(f"\nDetailed results saved in: data/optimization_results/")
            print("\nDemo completed successfully!")
            
        else:
            print("Analysis failed - no valid parameter combinations found.")
            print("This might happen with limited data. Try adjusting parameter ranges.")
    
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()