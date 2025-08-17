#!/usr/bin/env python3
"""
Test script for running strategy backtest using CSV data instead of API
"""
import pandas as pd
import yaml
from strategy.smc_fvg_loose_strategy import SMCFVGLooseStrategy
from backtest.backtester import Backtester
from utils.logger import get_logger
import logging
import os

class CSVBroker:
    """Mock broker that reads data from CSV file"""
    def __init__(self, csv_file, logger):
        self.csv_file = csv_file
        self.logger = logger
        
    def get_historical_klines(self, symbol, timeframe, start, end):
        """Load data from CSV file"""
        try:
            data = pd.read_csv(self.csv_file, parse_dates=['timestamp'])
            data.set_index('timestamp', inplace=True)
            
            # Filter by date range if specified
            if start:
                start_date = pd.to_datetime(start)
                data = data[data.index >= start_date]
            if end:
                end_date = pd.to_datetime(end)
                data = data[data.index <= end_date]
                
            self.logger.info(f"Loaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")
            return data[["open", "high", "low", "close", "volume"]].astype(float)
        except Exception as e:
            self.logger.error(f"Error loading CSV data: {e}")
            return None

def load_config():
    """Load configuration with CSV broker"""
    config = {
        "mode": "backtest",
        "strategy": {
            "name": "smc_fvg_loose",
            "symbol": "BTCUSDT",
            "timeframe": "4h",
            "position_size": 0.001,
            "max_positions": 1
        },
        "backtest": {
            "start": "2025-05-25",
            "end": "2025-08-01"
        },
        "risk": {
            "max_daily_loss": 100,
            "max_trades_per_day": 5
        },
        "logging": {
            "level": "INFO",
            "file": "data/trading.log"
        },
        "fee_rate": 0.001,        # 0.1% spot fee
        "slippage_rate": 0.0005,  # 0.05% slippage
        "min_order_size": 0.001,
        "in_sample_pct": 0.7,
        "initial_balance": 10000,
        "stop_loss_pct": 0.01,    # 1% stop loss
        "take_profit_pct": 0.02,  # 2% take profit
        "imbalance_threshold": 0.0002
    }
    return config

def main():
    config = load_config()
    
    # Setup logging to console as well
    logger = logging.getLogger("CryptoSMCAlgo")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Use CSV broker instead of Binance API
    broker = CSVBroker("BTCUSDT_4h_ohlcv.csv", logger)
    
    # Create strategy
    strategy = SMCFVGLooseStrategy(config["strategy"], broker)
    
    # Run backtest
    logger.info("Starting baseline backtest with current SMCFVGLooseStrategy")
    backtester = Backtester(config, strategy, logger)
    backtester.run()

if __name__ == "__main__":
    main()