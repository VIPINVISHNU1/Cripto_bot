#!/usr/bin/env python3
"""
Debug parameter optimization to find the exact issue.
"""

import yaml
import sys
import os
import pandas as pd
import traceback

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy.smc_fvg_loose_strategy import SMCFVGLooseStrategy
from utils.logger import get_logger
from backtest.backtester import Backtester

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

def debug_single_backtest():
    """Debug a single backtest run to understand the issue."""
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    logger = get_logger(config["logging"])
    
    # Use available data range
    config["backtest"]["start"] = "2025-05-25"
    config["backtest"]["end"] = "2025-08-16"
    
    # Initialize components
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
    
    print("Getting data...")
    data = broker.get_historical_klines("BTCUSDT", "4h", "2025-05-25", "2025-08-16")
    print(f"Data shape: {data.shape}")
    print(f"Data index type: {type(data.index)}")
    print(f"Data head:\n{data.head()}")
    
    # Split data for testing
    split_idx = int(len(data) * 0.7)
    data_in_sample = data.iloc[:split_idx]
    data_out_sample = data.iloc[split_idx:]
    
    print(f"In-sample: {len(data_in_sample)}, Out-sample: {len(data_out_sample)}")
    
    # Test single parameter set
    test_params = {
        'stop_loss_pct': 0.01,
        'take_profit_pct': 0.02,
        'position_size': 0.001,
        'imbalance_threshold': 0.0002
    }
    
    print(f"Testing parameters: {test_params}")
    
    try:
        # Create modified config
        modified_config = config.copy()
        modified_config.update(test_params)
        
        # Update strategy
        if hasattr(strategy, 'imbalance_threshold'):
            strategy.imbalance_threshold = test_params['imbalance_threshold']
        
        # Create backtester
        backtester = Backtester(modified_config, strategy, logger)
        backtester.stop_loss_pct = test_params['stop_loss_pct']
        backtester.take_profit_pct = test_params['take_profit_pct']
        backtester.position_size = test_params['position_size']
        
        print("Running in-sample backtest...")
        results_in = backtester.simulate_trades(data_in_sample, "debug_in")
        print(f"In-sample results: {len(results_in['trades'])} trades")
        print(f"Equity curve length: {len(results_in['equity_curve'])}")
        print(f"Final balance: {results_in['final_balance']}")
        
        print("Running out-sample backtest...")
        results_out = backtester.simulate_trades(data_out_sample, "debug_out")
        print(f"Out-sample results: {len(results_out['trades'])} trades")
        print(f"Equity curve length: {len(results_out['equity_curve'])}")
        print(f"Final balance: {results_out['final_balance']}")
        
        print("Backtest completed successfully!")
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_single_backtest()