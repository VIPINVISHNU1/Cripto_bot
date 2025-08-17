#!/usr/bin/env python3
"""
Test script to compare baseline vs enhanced strategy
"""
import pandas as pd
from strategy.smc_fvg_loose_strategy import SMCFVGLooseStrategy
from strategy.smc_fvg_enhanced_strategy import SMCFVGEnhancedStrategy
from backtest.backtester import Backtester
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

def get_enhanced_config():
    """Configuration for enhanced strategy"""
    config = {
        "mode": "backtest",
        "strategy": {
            "name": "smc_fvg_enhanced",
            "symbol": "BTCUSDT",
            "timeframe": "4h",
            "position_size": 0.001,
            "max_positions": 1,
            # Enhanced parameters
            "rsi_period": 14,
            "ema_period": 50,
            "atr_period": 14,
            "volume_period": 20,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "volume_multiplier": 1.2,
            "atr_multiplier_sl": 2.0,
            "atr_multiplier_tp": 3.0,
            "imbalance_threshold": 0.0002
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
        "stop_loss_pct": 0.01,    # 1% stop loss (fallback)
        "take_profit_pct": 0.02,  # 2% take profit (fallback)
    }
    return config

def get_baseline_config():
    """Configuration for baseline strategy"""
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
        "fee_rate": 0.001,
        "slippage_rate": 0.0005,
        "min_order_size": 0.001,
        "in_sample_pct": 0.7,
        "initial_balance": 10000,
        "stop_loss_pct": 0.01,
        "take_profit_pct": 0.02,
        "imbalance_threshold": 0.0002
    }
    return config

def setup_logger():
    """Setup console logger"""
    logger = logging.getLogger("CryptoSMCAlgo")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def run_comparison():
    """Run comparison between baseline and enhanced strategies"""
    logger = setup_logger()
    broker = CSVBroker("BTCUSDT_4h_ohlcv.csv", logger)
    
    print("="*80)
    print("BASELINE STRATEGY (SMCFVGLooseStrategy)")
    print("="*80)
    
    # Baseline strategy
    baseline_config = get_baseline_config()
    baseline_strategy = SMCFVGLooseStrategy(baseline_config["strategy"], broker)
    baseline_backtester = Backtester(baseline_config, baseline_strategy, logger)
    baseline_backtester.run()
    
    print("\n" + "="*80)
    print("ENHANCED STRATEGY (SMCFVGEnhancedStrategy)")
    print("="*80)
    
    # Enhanced strategy
    enhanced_config = get_enhanced_config()
    enhanced_strategy = SMCFVGEnhancedStrategy(enhanced_config["strategy"], broker)
    enhanced_backtester = Backtester(enhanced_config, enhanced_strategy, logger)
    enhanced_backtester.run()

if __name__ == "__main__":
    run_comparison()