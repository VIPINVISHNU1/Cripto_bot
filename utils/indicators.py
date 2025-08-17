import pandas as pd
import numpy as np
import talib

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    return talib.RSI(data['close'].values, timeperiod=period)

def calculate_ema(data, period=20):
    """Calculate EMA indicator"""
    return talib.EMA(data['close'].values, timeperiod=period)

def calculate_atr(data, period=14):
    """Calculate ATR indicator"""
    return talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)

def calculate_adx(data, period=14):
    """Calculate ADX for trend strength"""
    return talib.ADX(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)

def calculate_volume_sma(data, period=20):
    """Calculate volume SMA for volume confirmation"""
    return talib.SMA(data['volume'].values, timeperiod=period)

def detect_market_structure(data, lookback=20):
    """Detect market structure (Higher Highs, Lower Lows, etc.)"""
    highs = data['high'].rolling(window=lookback).max()
    lows = data['low'].rolling(window=lookback).min()
    
    # Simple trend detection
    trend = np.where(data['close'] > highs.shift(1), 1,  # Bullish
                    np.where(data['close'] < lows.shift(1), -1, 0))  # Bearish or Neutral
    
    return trend

def detect_order_blocks(data, strength=3):
    """Detect order blocks - institutional footprints"""
    order_blocks = []
    
    for i in range(strength, len(data) - strength):
        current_bar = data.iloc[i]
        
        # Bullish order block: Strong bullish candle followed by continuation
        is_bullish_ob = (
            current_bar['close'] > current_bar['open'] and  # Bullish candle
            (current_bar['high'] - current_bar['low']) > 2 * abs(current_bar['close'] - current_bar['open']) and  # Strong volume
            current_bar['volume'] > data['volume'].iloc[i-strength:i+strength].mean() * 1.5  # High volume
        )
        
        # Bearish order block: Strong bearish candle followed by continuation  
        is_bearish_ob = (
            current_bar['close'] < current_bar['open'] and  # Bearish candle
            (current_bar['high'] - current_bar['low']) > 2 * abs(current_bar['close'] - current_bar['open']) and  # Strong volume
            current_bar['volume'] > data['volume'].iloc[i-strength:i+strength].mean() * 1.5  # High volume
        )
        
        if is_bullish_ob:
            order_blocks.append({
                'type': 'bullish',
                'index': i,
                'high': current_bar['high'],
                'low': current_bar['low'],
                'timestamp': data.index[i]
            })
        elif is_bearish_ob:
            order_blocks.append({
                'type': 'bearish', 
                'index': i,
                'high': current_bar['high'],
                'low': current_bar['low'],
                'timestamp': data.index[i]
            })
    
    return order_blocks

def detect_liquidity_sweeps(data, lookback=10):
    """Detect liquidity sweeps - stop hunts"""
    sweeps = []
    
    for i in range(lookback, len(data)):
        current_high = data['high'].iloc[i]
        current_low = data['low'].iloc[i]
        
        # Recent highs/lows
        recent_highs = data['high'].iloc[i-lookback:i]
        recent_lows = data['low'].iloc[i-lookback:i]
        
        # Liquidity sweep above recent highs
        if current_high > recent_highs.max() * 1.001:  # 0.1% buffer
            sweeps.append({
                'type': 'high_sweep',
                'index': i,
                'price': current_high,
                'timestamp': data.index[i]
            })
        
        # Liquidity sweep below recent lows  
        if current_low < recent_lows.min() * 0.999:  # 0.1% buffer
            sweeps.append({
                'type': 'low_sweep', 
                'index': i,
                'price': current_low,
                'timestamp': data.index[i]
            })
    
    return sweeps