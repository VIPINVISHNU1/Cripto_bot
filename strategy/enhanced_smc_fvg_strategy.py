import pandas as pd
import numpy as np
from utils.indicators import (
    calculate_rsi, calculate_ema, calculate_atr, calculate_adx,
    calculate_volume_sma, detect_market_structure, detect_order_blocks,
    detect_liquidity_sweeps
)

class EnhancedSMCFVGStrategy:
    """Enhanced SMC FVG Strategy with confluence factors and smart money concepts"""
    
    def __init__(self, config, broker=None):
        self.config = config
        self.broker = broker
        
        # Strategy parameters
        self.rsi_period = 14
        self.ema_fast_period = 20
        self.ema_slow_period = 50
        self.atr_period = 14
        self.adx_period = 14
        self.volume_period = 20
        
        # Confluence thresholds
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.trend_strength_threshold = 25  # ADX threshold
        self.volume_multiplier = 1.2  # Volume above average
        
        # Risk management
        self.atr_stop_multiplier = 2.0
        self.atr_tp_multiplier = 3.0
        self.min_risk_reward = 2.0
        
    def add_indicators(self, data):
        """Add all technical indicators to the data"""
        data_copy = data.copy()
        
        # Momentum indicators
        data_copy['rsi'] = calculate_rsi(data_copy, self.rsi_period)
        
        # Trend indicators
        data_copy['ema_fast'] = calculate_ema(data_copy, self.ema_fast_period)
        data_copy['ema_slow'] = calculate_ema(data_copy, self.ema_slow_period)
        data_copy['adx'] = calculate_adx(data_copy, self.adx_period)
        
        # Volatility indicators
        data_copy['atr'] = calculate_atr(data_copy, self.atr_period)
        
        # Volume indicators
        data_copy['volume_sma'] = calculate_volume_sma(data_copy, self.volume_period)
        
        # Market structure
        data_copy['market_structure'] = detect_market_structure(data_copy)
        
        return data_copy

    def detect_enhanced_fvg(self, data):
        """Detect FVGs with enhanced validation"""
        fvg_list = []
        
        for i in range(2, len(data)):
            c1 = data.iloc[i-2]
            c2 = data.iloc[i-1]  # Middle candle
            c3 = data.iloc[i]
            
            # Basic FVG detection
            bullish_fvg = c1['low'] > c3['high']
            bearish_fvg = c1['high'] < c3['low']
            
            if bullish_fvg:
                # Calculate confluence score for bullish FVG
                score = self.calculate_confluence_score(data, i, 'bullish')
                
                if score >= 3:  # Minimum confluence threshold
                    fvg = {
                        "type": "bullish",
                        "upper": c3['high'],
                        "lower": c1['low'],
                        "created_idx": i,
                        "touched": False,
                        "confluence_score": score,
                        "atr": data['atr'].iloc[i],
                        "volume_ratio": data['volume'].iloc[i] / data['volume_sma'].iloc[i]
                    }
                    fvg_list.append(fvg)
                    
            elif bearish_fvg:
                # Calculate confluence score for bearish FVG
                score = self.calculate_confluence_score(data, i, 'bearish')
                
                if score >= 3:  # Minimum confluence threshold
                    fvg = {
                        "type": "bearish",
                        "upper": c1['high'],
                        "lower": c3['low'],
                        "created_idx": i,
                        "touched": False,
                        "confluence_score": score,
                        "atr": data['atr'].iloc[i],
                        "volume_ratio": data['volume'].iloc[i] / data['volume_sma'].iloc[i]
                    }
                    fvg_list.append(fvg)
        
        return fvg_list
    
    def calculate_confluence_score(self, data, idx, direction):
        """Calculate confluence score based on multiple factors"""
        score = 0
        bar = data.iloc[idx]
        
        # RSI confluence
        if direction == 'bullish' and bar['rsi'] < 50:  # Not overbought for bullish
            score += 1
        elif direction == 'bearish' and bar['rsi'] > 50:  # Not oversold for bearish
            score += 1
            
        # EMA trend confluence
        if direction == 'bullish' and bar['ema_fast'] > bar['ema_slow']:  # Uptrend
            score += 1
        elif direction == 'bearish' and bar['ema_fast'] < bar['ema_slow']:  # Downtrend
            score += 1
            
        # ADX trend strength
        if bar['adx'] > self.trend_strength_threshold:
            score += 1
            
        # Volume confirmation
        if bar['volume'] > bar['volume_sma'] * self.volume_multiplier:
            score += 1
            
        # Market structure alignment
        if direction == 'bullish' and bar['market_structure'] >= 0:
            score += 1
        elif direction == 'bearish' and bar['market_structure'] <= 0:
            score += 1
            
        return score
    
    def calculate_dynamic_targets(self, entry_price, direction, atr_value):
        """Calculate dynamic stop-loss and take-profit based on ATR"""
        if direction == 'long':
            stop_loss = entry_price - (atr_value * self.atr_stop_multiplier)
            take_profit = entry_price + (atr_value * self.atr_tp_multiplier)
        else:  # short
            stop_loss = entry_price + (atr_value * self.atr_stop_multiplier)
            take_profit = entry_price - (atr_value * self.atr_tp_multiplier)
            
        # Check risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if reward / risk < self.min_risk_reward:
            # Adjust take profit to meet minimum risk-reward
            if direction == 'long':
                take_profit = entry_price + (risk * self.min_risk_reward)
            else:
                take_profit = entry_price - (risk * self.min_risk_reward)
                
        return stop_loss, take_profit
    
    def run_backtest(self, data: pd.DataFrame):
        """Enhanced backtest with confluence and smart money concepts"""
        # Ensure data index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex (timestamp)!")
        
        # Add indicators
        enhanced_data = self.add_indicators(data)
        
        # Detect smart money concepts
        order_blocks = detect_order_blocks(enhanced_data)
        liquidity_sweeps = detect_liquidity_sweeps(enhanced_data)
        
        # Detect enhanced FVGs
        fvg_list = self.detect_enhanced_fvg(enhanced_data)
        
        trades = []
        
        # Entry logic with enhanced validation
        for fvg in fvg_list:
            idx_after_fvg = fvg["created_idx"] + 1
            if idx_after_fvg >= len(enhanced_data):
                continue
                
            for i in range(idx_after_fvg, len(enhanced_data)):
                bar = enhanced_data.iloc[i]
                if fvg["touched"]:
                    break
                
                # Check for FVG touch with additional validation
                if fvg["type"] == "bullish" and bar["low"] <= fvg["upper"]:
                    # Additional validation for entry
                    if self.validate_entry(enhanced_data, i, 'long', order_blocks, liquidity_sweeps):
                        # Calculate dynamic targets
                        entry_price = bar["close"]
                        stop_loss, take_profit = self.calculate_dynamic_targets(
                            entry_price, 'long', fvg["atr"]
                        )
                        
                        trades.append({
                            "time": enhanced_data.index[i],
                            "type": "long",
                            "reason": "Enhanced_FVG_touch",
                            "bar_index": i,
                            "confluence_score": fvg["confluence_score"],
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "atr": fvg["atr"],
                            "volume_ratio": fvg["volume_ratio"]
                        })
                        fvg["touched"] = True
                        
                elif fvg["type"] == "bearish" and bar["high"] >= fvg["lower"]:
                    # Additional validation for entry
                    if self.validate_entry(enhanced_data, i, 'short', order_blocks, liquidity_sweeps):
                        # Calculate dynamic targets
                        entry_price = bar["close"]
                        stop_loss, take_profit = self.calculate_dynamic_targets(
                            entry_price, 'short', fvg["atr"]
                        )
                        
                        trades.append({
                            "time": enhanced_data.index[i],
                            "type": "short", 
                            "reason": "Enhanced_FVG_touch",
                            "bar_index": i,
                            "confluence_score": fvg["confluence_score"],
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "atr": fvg["atr"],
                            "volume_ratio": fvg["volume_ratio"]
                        })
                        fvg["touched"] = True
        
        print(f"DEBUG: Enhanced FVGs detected: {len(fvg_list)}, Trades generated: {len(trades)}")
        print(f"DEBUG: Order blocks detected: {len(order_blocks)}, Liquidity sweeps: {len(liquidity_sweeps)}")
        
        return trades
    
    def validate_entry(self, data, idx, direction, order_blocks, liquidity_sweeps):
        """Additional entry validation using smart money concepts"""
        bar = data.iloc[idx]
        
        # Check for recent liquidity sweeps that might indicate institutional activity
        recent_sweeps = [s for s in liquidity_sweeps if abs(s['index'] - idx) <= 5]
        
        # Bullish validation
        if direction == 'long':
            # Look for low sweeps (liquidity grab) before bullish move
            low_sweeps = [s for s in recent_sweeps if s['type'] == 'low_sweep']
            if low_sweeps:
                return True
                
            # Check for bullish order blocks nearby
            nearby_obs = [ob for ob in order_blocks 
                         if ob['type'] == 'bullish' and abs(ob['index'] - idx) <= 10]
            if nearby_obs:
                return True
                
        # Bearish validation  
        elif direction == 'short':
            # Look for high sweeps (liquidity grab) before bearish move
            high_sweeps = [s for s in recent_sweeps if s['type'] == 'high_sweep']
            if high_sweeps:
                return True
                
            # Check for bearish order blocks nearby
            nearby_obs = [ob for ob in order_blocks 
                         if ob['type'] == 'bearish' and abs(ob['index'] - idx) <= 10]
            if nearby_obs:
                return True
        
        # Default validation based on trend and momentum
        if (direction == 'long' and bar['rsi'] < 60 and bar['ema_fast'] > bar['ema_slow']):
            return True
        elif (direction == 'short' and bar['rsi'] > 40 and bar['ema_fast'] < bar['ema_slow']):
            return True
            
        return False