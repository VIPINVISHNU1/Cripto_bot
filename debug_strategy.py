#!/usr/bin/env python3
"""
Debug script to analyze trade generation and validation
"""
import pandas as pd
import numpy as np
from strategy.smc_fvg_enhanced_strategy import SMCFVGEnhancedStrategy
import logging

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
                
            return data[["open", "high", "low", "close", "volume"]].astype(float)
        except Exception as e:
            self.logger.error(f"Error loading CSV data: {e}")
            return None

def debug_strategy():
    """Debug the enhanced strategy to understand trade generation"""
    
    # Setup
    logger = logging.getLogger("Debug")
    logger.setLevel(logging.INFO)
    broker = CSVBroker("BTCUSDT_4h_ohlcv.csv", logger)
    
    # Load data
    data = broker.get_historical_klines("BTCUSDT", "4h", "2025-05-25", "2025-08-01")
    split_idx = int(len(data) * 0.7)
    data_in_sample = data.iloc[:split_idx]
    
    # Balanced configuration - not too restrictive
    config = {
        "rsi_period": 14,
        "ema_period": 50,
        "atr_period": 14,
        "volume_period": 20,
        "rsi_oversold": 35,      # Less restrictive
        "rsi_overbought": 65,    # Less restrictive
        "volume_multiplier": 1.1, # Lower volume requirement
        "atr_multiplier_sl": 2.0,
        "atr_multiplier_tp": 3.0,
        "imbalance_threshold": 0.0002
    }
    
    # Create strategy and manually call methods for debugging
    strategy = SMCFVGEnhancedStrategy(config, broker)
    
    # Calculate indicators
    data_with_indicators = strategy.calculate_indicators(data_in_sample)
    
    print(f"Data shape: {data_with_indicators.shape}")
    print(f"Columns: {data_with_indicators.columns.tolist()}")
    print("\nFirst few rows with indicators:")
    print(data_with_indicators[['close', 'rsi', 'ema', 'atr', 'volume_ma']].head(10))
    
    # Detect FVGs
    fvg_list = []
    for i in range(2, len(data_with_indicators)):
        fvg = strategy.detect_fvg(data_with_indicators, i)
        if fvg:
            fvg_list.append(fvg)
    
    print(f"\nFVGs detected: {len(fvg_list)}")
    if fvg_list:
        print("Sample FVGs:")
        for i, fvg in enumerate(fvg_list[:5]):
            print(f"  {i+1}: {fvg}")
    
    # Test validation for some FVGs
    validated_count = 0
    rejected_count = 0
    validation_reasons = {}
    
    for fvg in fvg_list[:10]:  # Test first 10 FVGs
        idx_after_fvg = fvg["created_idx"] + 1
        if idx_after_fvg >= len(data_with_indicators):
            continue
            
        for i in range(idx_after_fvg, min(idx_after_fvg + 5, len(data_with_indicators))):  # Check next 5 bars
            bar = data_with_indicators.iloc[i]
            
            # Check for touch
            if fvg["type"] == "bullish" and bar["low"] <= fvg["upper"]:
                is_valid, validation_info = strategy.validate_signal(data_with_indicators, i, "bullish")
                if is_valid:
                    validated_count += 1
                    print(f"\nVALIDATED - Bullish FVG at {data_with_indicators.index[i]}")
                    print(f"  Validation: {validation_info}")
                else:
                    rejected_count += 1
                    for reason in validation_info.get('reasons', []):
                        validation_reasons[reason] = validation_reasons.get(reason, 0) + 1
                break
                
            elif fvg["type"] == "bearish" and bar["high"] >= fvg["lower"]:
                is_valid, validation_info = strategy.validate_signal(data_with_indicators, i, "bearish")
                if is_valid:
                    validated_count += 1
                    print(f"\nVALIDATED - Bearish FVG at {data_with_indicators.index[i]}")
                    print(f"  Validation: {validation_info}")
                else:
                    rejected_count += 1
                    for reason in validation_info.get('reasons', []):
                        validation_reasons[reason] = validation_reasons.get(reason, 0) + 1
                break
    
    print(f"\nValidation Summary (first 10 FVGs):")
    print(f"  Validated: {validated_count}")
    print(f"  Rejected: {rejected_count}")
    print(f"  Validation reasons: {validation_reasons}")
    
    # Check some indicator stats
    print(f"\nIndicator Statistics:")
    print(f"  RSI range: {data_with_indicators['rsi'].min():.1f} - {data_with_indicators['rsi'].max():.1f}")
    print(f"  Volume ratio range: {(data_with_indicators['volume']/data_with_indicators['volume_ma']).min():.2f} - {(data_with_indicators['volume']/data_with_indicators['volume_ma']).max():.2f}")
    print(f"  ATR percentage range: {(data_with_indicators['atr']/data_with_indicators['close']*100).min():.2f}% - {(data_with_indicators['atr']/data_with_indicators['close']*100).max():.2f}%")

if __name__ == "__main__":
    debug_strategy()