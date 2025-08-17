import pandas as pd
import numpy as np
from utils.indicators import (
    calculate_rsi, calculate_ema, calculate_atr, calculate_adx,
    calculate_volume_sma, detect_market_structure, detect_order_blocks,
    detect_liquidity_sweeps
)

class AdvancedSMCFVGStrategy:
    """Advanced SMC FVG Strategy with strict filtering and market regime detection"""
    
    def __init__(self, config, broker=None):
        self.config = config
        self.broker = broker
        
        # Strategy parameters (more conservative)
        self.rsi_period = 14
        self.ema_fast_period = 21
        self.ema_slow_period = 50
        self.atr_period = 14
        self.adx_period = 14
        self.volume_period = 20
        
        # Confluence thresholds (less strict but still selective)
        self.rsi_oversold = 40
        self.rsi_overbought = 60
        self.trend_strength_threshold = 20  # Lower ADX threshold
        self.volume_multiplier = 1.2        # Lower volume requirement
        self.min_confluence_score = 3       # Lower but still meaningful threshold
        
        # Enhanced risk management
        self.atr_stop_multiplier = 1.5  # Tighter stops
        self.atr_tp_multiplier = 4.0   # Larger targets
        self.min_risk_reward = 2.5     # Better risk-reward
        
        # Market regime parameters (less strict)
        self.trending_adx_threshold = 20
        self.ranging_adx_threshold = 15
        
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
        
        # Market regime detection
        data_copy['market_regime'] = self.detect_market_regime(data_copy)
        
        return data_copy
    
    def detect_market_regime(self, data):
        """Detect market regime: trending (1), ranging (0), or uncertain (-1)"""
        regime = np.zeros(len(data))
        
        for i in range(self.adx_period, len(data)):
            adx_val = data['adx'].iloc[i]
            ema_diff = abs(data['ema_fast'].iloc[i] - data['ema_slow'].iloc[i])
            price_range = data['close'].iloc[i-10:i].max() - data['close'].iloc[i-10:i].min()
            avg_price = data['close'].iloc[i-10:i].mean()
            range_pct = price_range / avg_price if avg_price > 0 else 0
            
            if adx_val > self.trending_adx_threshold and range_pct > 0.02:
                regime[i] = 1  # Trending
            elif adx_val < self.ranging_adx_threshold:
                regime[i] = 0  # Ranging
            else:
                regime[i] = -1  # Uncertain
                
        return regime

    def detect_high_quality_fvg(self, data):
        """Detect only highest quality FVGs with strict filtering"""
        fvg_list = []
        
        for i in range(max(25, self.ema_slow_period), len(data)):  # Start after indicators stabilize
            c1 = data.iloc[i-2]
            c2 = data.iloc[i-1]
            c3 = data.iloc[i]
            
            # Basic FVG detection
            bullish_fvg = c1['low'] > c3['high']
            bearish_fvg = c1['high'] < c3['low']
            
            # Allow trending and uncertain markets (but not ranging)
            if data['market_regime'].iloc[i] == 0:  # Skip only ranging markets
                continue
                
            if bullish_fvg:
                score = self.calculate_strict_confluence_score(data, i, 'bullish')
                
                if score >= self.min_confluence_score:
                    # Additional quality checks
                    if self.validate_fvg_quality(data, i, 'bullish'):
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
                score = self.calculate_strict_confluence_score(data, i, 'bearish')
                
                if score >= self.min_confluence_score:
                    # Additional quality checks
                    if self.validate_fvg_quality(data, i, 'bearish'):
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
    
    def calculate_strict_confluence_score(self, data, idx, direction):
        """Calculate strict confluence score with higher standards"""
        score = 0
        bar = data.iloc[idx]
        
        # RSI confluence (stricter)
        if direction == 'bullish':
            if bar['rsi'] < self.rsi_overbought and bar['rsi'] > self.rsi_oversold:
                score += 1
            if bar['rsi'] < 45:  # Additional bonus for oversold areas
                score += 1
        elif direction == 'bearish':
            if bar['rsi'] > self.rsi_oversold and bar['rsi'] < self.rsi_overbought:
                score += 1
            if bar['rsi'] > 55:  # Additional bonus for overbought areas
                score += 1
        
        # Strong EMA trend confluence
        ema_diff = abs(bar['ema_fast'] - bar['ema_slow'])
        price_threshold = bar['close'] * 0.005  # 0.5% minimum difference
        
        if ema_diff > price_threshold:
            if direction == 'bullish' and bar['ema_fast'] > bar['ema_slow']:
                score += 2  # Double weight for strong trend
            elif direction == 'bearish' and bar['ema_fast'] < bar['ema_slow']:
                score += 2
        
        # ADX trend strength (stricter)
        if bar['adx'] > self.trend_strength_threshold:
            score += 1
            if bar['adx'] > 35:  # Very strong trend
                score += 1
        
        # High volume confirmation
        volume_ratio = bar['volume'] / bar['volume_sma']
        if volume_ratio > self.volume_multiplier:
            score += 1
            if volume_ratio > 2.0:  # Exceptional volume
                score += 1
        
        # Market structure alignment (stricter)
        if direction == 'bullish' and bar['market_structure'] > 0:
            score += 1
        elif direction == 'bearish' and bar['market_structure'] < 0:
            score += 1
        
        # Price action confirmation
        candle = data.iloc[idx-1:idx+1]  # 3-candle pattern
        if self.validate_price_action(candle, direction):
            score += 1
            
        return score
    
    def validate_fvg_quality(self, data, idx, direction):
        """Additional FVG quality validation"""
        # Check FVG size relative to ATR
        if direction == 'bullish':
            fvg_size = data.iloc[idx-2]['low'] - data.iloc[idx]['high']
        else:
            fvg_size = data.iloc[idx-2]['high'] - data.iloc[idx]['low']
            
        atr_val = data['atr'].iloc[idx]
        
        # FVG should be significant but not too large
        if fvg_size < atr_val * 0.3 or fvg_size > atr_val * 2.0:
            return False
            
        # Check for clean price action around FVG
        lookback_bars = data.iloc[max(0, idx-10):idx]
        
        if direction == 'bullish':
            # No major resistance nearby
            resistance_level = data.iloc[idx]['high']
            recent_highs = lookback_bars['high'].max()
            if recent_highs > resistance_level * 1.01:  # 1% buffer
                return False
        else:
            # No major support nearby
            support_level = data.iloc[idx]['low']
            recent_lows = lookback_bars['low'].min()
            if recent_lows < support_level * 0.99:  # 1% buffer
                return False
                
        return True
    
    def validate_price_action(self, candles, direction):
        """Validate price action pattern"""
        if len(candles) < 2:
            return False
            
        if direction == 'bullish':
            # Look for bullish momentum
            return (candles['close'].iloc[-1] > candles['open'].iloc[-1] and
                   candles['close'].iloc[-1] > candles['close'].iloc[-2])
        else:
            # Look for bearish momentum
            return (candles['close'].iloc[-1] < candles['open'].iloc[-1] and
                   candles['close'].iloc[-1] < candles['close'].iloc[-2])
    
    def calculate_dynamic_targets(self, entry_price, direction, atr_value):
        """Calculate dynamic stop-loss and take-profit based on ATR"""
        if direction == 'long':
            stop_loss = entry_price - (atr_value * self.atr_stop_multiplier)
            take_profit = entry_price + (atr_value * self.atr_tp_multiplier)
        else:
            stop_loss = entry_price + (atr_value * self.atr_stop_multiplier)
            take_profit = entry_price - (atr_value * self.atr_tp_multiplier)
            
        # Ensure minimum risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if reward / risk < self.min_risk_reward:
            if direction == 'long':
                take_profit = entry_price + (risk * self.min_risk_reward)
            else:
                take_profit = entry_price - (risk * self.min_risk_reward)
                
        return stop_loss, take_profit
    
    def run_backtest(self, data: pd.DataFrame):
        """Advanced backtest with strict filtering"""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex (timestamp)!")
        
        # Add indicators
        enhanced_data = self.add_indicators(data)
        
        # Detect smart money concepts
        order_blocks = detect_order_blocks(enhanced_data)
        liquidity_sweeps = detect_liquidity_sweeps(enhanced_data)
        
        # Detect high-quality FVGs only
        fvg_list = self.detect_high_quality_fvg(enhanced_data)
        
        trades = []
        
        # Entry logic with very strict validation
        for fvg in fvg_list:
            idx_after_fvg = fvg["created_idx"] + 1
            if idx_after_fvg >= len(enhanced_data):
                continue
                
            # Limit search window to prevent stale signals
            max_search_bars = 20
            end_idx = min(idx_after_fvg + max_search_bars, len(enhanced_data))
                
            for i in range(idx_after_fvg, end_idx):
                bar = enhanced_data.iloc[i]
                if fvg["touched"]:
                    break
                
                # Check market regime at entry time
                if enhanced_data['market_regime'].iloc[i] != 1:  # Only trending markets
                    continue
                
                # FVG touch validation
                if fvg["type"] == "bullish" and bar["low"] <= fvg["upper"]:
                    if self.validate_advanced_entry(enhanced_data, i, 'long', order_blocks, liquidity_sweeps):
                        entry_price = min(bar["close"], fvg["upper"])  # Better fill simulation
                        stop_loss, take_profit = self.calculate_dynamic_targets(
                            entry_price, 'long', fvg["atr"]
                        )
                        
                        trades.append({
                            "time": enhanced_data.index[i],
                            "type": "long",
                            "reason": "Advanced_FVG_touch",
                            "bar_index": i,
                            "confluence_score": fvg["confluence_score"],
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "atr": fvg["atr"],
                            "volume_ratio": fvg["volume_ratio"],
                            "market_regime": enhanced_data['market_regime'].iloc[i]
                        })
                        fvg["touched"] = True
                        
                elif fvg["type"] == "bearish" and bar["high"] >= fvg["lower"]:
                    if self.validate_advanced_entry(enhanced_data, i, 'short', order_blocks, liquidity_sweeps):
                        entry_price = max(bar["close"], fvg["lower"])  # Better fill simulation
                        stop_loss, take_profit = self.calculate_dynamic_targets(
                            entry_price, 'short', fvg["atr"]
                        )
                        
                        trades.append({
                            "time": enhanced_data.index[i],
                            "type": "short",
                            "reason": "Advanced_FVG_touch",
                            "bar_index": i,
                            "confluence_score": fvg["confluence_score"],
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "atr": fvg["atr"],
                            "volume_ratio": fvg["volume_ratio"],
                            "market_regime": enhanced_data['market_regime'].iloc[i]
                        })
                        fvg["touched"] = True
        
        print(f"DEBUG: Advanced FVGs detected: {len(fvg_list)}, Trades generated: {len(trades)}")
        print(f"DEBUG: Order blocks detected: {len(order_blocks)}, Liquidity sweeps: {len(liquidity_sweeps)}")
        
        # Print average confluence score
        if fvg_list:
            avg_confluence = sum(fvg["confluence_score"] for fvg in fvg_list) / len(fvg_list)
            print(f"DEBUG: Average confluence score: {avg_confluence:.2f}")
        
        return trades
    
    def validate_advanced_entry(self, data, idx, direction, order_blocks, liquidity_sweeps):
        """Advanced entry validation with multiple checks"""
        bar = data.iloc[idx]
        
        # Current market conditions
        current_rsi = bar['rsi']
        current_adx = bar['adx']
        volume_ratio = bar['volume'] / bar['volume_sma']
        
        # Basic momentum and volume checks
        if current_adx < self.trending_adx_threshold:
            return False
            
        if volume_ratio < 1.0:  # Below average volume
            return False
        
        # Direction-specific validations
        if direction == 'long':
            # For longs: RSI not too high, bullish momentum
            if current_rsi > 70:
                return False
            if bar['ema_fast'] <= bar['ema_slow']:
                return False
                
            # Check for supportive liquidity sweeps
            recent_sweeps = [s for s in liquidity_sweeps 
                           if s['type'] == 'low_sweep' and abs(s['index'] - idx) <= 3]
            if recent_sweeps:
                return True
                
        elif direction == 'short':
            # For shorts: RSI not too low, bearish momentum
            if current_rsi < 30:
                return False
            if bar['ema_fast'] >= bar['ema_slow']:
                return False
                
            # Check for supportive liquidity sweeps
            recent_sweeps = [s for s in liquidity_sweeps 
                           if s['type'] == 'high_sweep' and abs(s['index'] - idx) <= 3]
            if recent_sweeps:
                return True
        
        # Check for supportive order blocks
        nearby_obs = [ob for ob in order_blocks 
                     if ob['type'] == ('bullish' if direction == 'long' else 'bearish') 
                     and abs(ob['index'] - idx) <= 5]
        
        if nearby_obs:
            return True
        
        # Final validation: strong momentum confirmation
        momentum_bars = data.iloc[max(0, idx-3):idx+1]
        if direction == 'long':
            bullish_momentum = (momentum_bars['close'] > momentum_bars['open']).sum() >= 2
            return bullish_momentum
        else:
            bearish_momentum = (momentum_bars['close'] < momentum_bars['open']).sum() >= 2
            return bearish_momentum