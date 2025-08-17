#!/usr/bin/env python3
"""
Final optimized test for the enhanced strategy
"""
import pandas as pd
from strategy.smc_fvg_enhanced_strategy import SMCFVGEnhancedStrategy
from backtest.enhanced_backtester import EnhancedBacktester
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

def get_optimized_config():
    """Optimized configuration for best balance"""
    config = {
        "mode": "backtest",
        "strategy": {
            "name": "smc_fvg_optimized",
            "symbol": "BTCUSDT",
            "timeframe": "4h",
            "position_size": 0.001,
            "max_positions": 1,
            # Optimized parameters
            "rsi_period": 14,
            "ema_period": 50,
            "atr_period": 14,
            "volume_period": 20,
            "rsi_oversold": 30,      
            "rsi_overbought": 70,    
            "volume_multiplier": 1.0, # No volume requirement for now
            "atr_multiplier_sl": 2.0, 
            "atr_multiplier_tp": 3.0, 
            "imbalance_threshold": 0.0003  # Slightly larger gaps
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
        "account_risk_pct": 0.005, # Reduce to 0.5% account risk per trade
        "max_position_size": 0.05, # Max 5% of account per trade
        "stop_loss_pct": 0.015,   # 1.5% fallback stop loss
        "take_profit_pct": 0.03,  # 3% fallback take profit
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

def main():
    """Test optimized strategy"""
    logger = setup_logger()
    broker = CSVBroker("BTCUSDT_4h_ohlcv.csv", logger)
    
    print("="*80)
    print("OPTIMIZED ENHANCED STRATEGY")
    print("="*80)
    
    # Optimized strategy
    config = get_optimized_config()
    strategy = SMCFVGEnhancedStrategy(config["strategy"], broker)
    backtester = EnhancedBacktester(config, strategy, logger)
    backtester.run()

if __name__ == "__main__":
    main()