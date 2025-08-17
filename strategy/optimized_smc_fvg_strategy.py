import pandas as pd
import numpy as np
from utils.indicators import (
    calculate_rsi, calculate_ema, calculate_atr, calculate_adx,
    calculate_volume_sma, detect_market_structure, detect_order_blocks,
    detect_liquidity_sweeps
)

class OptimizedSMCFVGStrategy:
    """Optimized SMC FVG Strategy balancing quality and quantity for 80% win rate"""
    
    def __init__(self, config, broker=None):
        self.config = config
        self.broker = broker
        
        # Optimized parameters for better balance
        self.rsi_period = 14
        self.ema_fast_period = 21
        self.ema_slow_period = 50
        self.atr_period = 14
        self.adx_period = 14
        self.volume_period = 20
        
        # More permissive balanced thresholds
        self.rsi_oversold = 50
        self.rsi_overbought = 50
        self.trend_strength_threshold = 15
        self.volume_multiplier = 1.0
        self.min_confluence_score = 3
        
        # Conservative risk management for higher win rate
        self.atr_stop_multiplier = 1.2   # Tighter stops
        self.atr_tp_multiplier = 3.5     # Good risk-reward
        self.min_risk_reward = 2.0
        
        # More permissive market regime
        self.trending_adx_threshold = 15
        self.ranging_adx_threshold = 10
        
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
        
        # Add momentum confirmation
        data_copy['momentum_score'] = self.calculate_momentum_score(data_copy)
        
        return data_copy
    
    def detect_market_regime(self, data):
        """Detect market regime with balanced approach"""
        regime = np.zeros(len(data))
        
        for i in range(self.adx_period, len(data)):
            adx_val = data['adx'].iloc[i]
            
            # Price momentum over short and medium term
            short_momentum = data['close'].iloc[i] - data['close'].iloc[i-5]
            medium_momentum = data['close'].iloc[i] - data['close'].iloc[i-10]
            
            if adx_val > self.trending_adx_threshold and (short_momentum * medium_momentum > 0):
                regime[i] = 1  # Trending
            elif adx_val < self.ranging_adx_threshold:
                regime[i] = 0  # Ranging
            else:
                regime[i] = -1  # Uncertain but tradeable
                
        return regime
    
    def calculate_momentum_score(self, data):
        """Calculate momentum score for additional confirmation"""
        momentum_score = np.zeros(len(data))
        
        for i in range(10, len(data)):
            score = 0
            
            # Price momentum
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                score += 1
            if data['close'].iloc[i] > data['close'].iloc[i-3]:
                score += 1
            if data['close'].iloc[i] > data['close'].iloc[i-5]:
                score += 1
                
            # EMA momentum
            if data['ema_fast'].iloc[i] > data['ema_fast'].iloc[i-1]:
                score += 1
                
            # Volume momentum
            if data['volume'].iloc[i] > data['volume'].iloc[i-1]:
                score += 1
                
            momentum_score[i] = score
            
        return momentum_score

    def detect_quality_fvg(self, data):
        """Detect quality FVGs with balanced filtering"""
        fvg_list = []
        
        print(f"DEBUG: Starting FVG detection from index {max(25, self.ema_slow_period)} to {len(data)}")
        
        for i in range(max(25, self.ema_slow_period), len(data)):
            c1 = data.iloc[i-2]
            c2 = data.iloc[i-1]
            c3 = data.iloc[i]
            
            # Basic FVG detection
            bullish_fvg = c1['low'] > c3['high']
            bearish_fvg = c1['high'] < c3['low']
            
            if bullish_fvg or bearish_fvg:
                print(f"DEBUG: Found {'bullish' if bullish_fvg else 'bearish'} FVG at index {i}")
                
                # Check if indicators are available
                if pd.isna(data['rsi'].iloc[i]) or pd.isna(data['ema_fast'].iloc[i]):
                    print(f"DEBUG: Skipping FVG at {i} due to missing indicators")
                    continue
                
                direction = 'bullish' if bullish_fvg else 'bearish'
                score = self.calculate_balanced_confluence_score(data, i, direction)
                print(f"DEBUG: FVG at {i} has confluence score: {score}")
                
                if score >= self.min_confluence_score:
                    context_valid = self.validate_fvg_context(data, i, direction)
                    print(f"DEBUG: Context validation for FVG at {i}: {context_valid}")
                    
                    if context_valid:
                        if direction == 'bullish':
                            fvg = {
                                "type": "bullish",
                                "upper": c3['high'],
                                "lower": c1['low'],
                                "created_idx": i,
                                "touched": False,
                                "confluence_score": score,
                                "atr": data['atr'].iloc[i],
                                "volume_ratio": data['volume'].iloc[i] / data['volume_sma'].iloc[i],
                                "momentum_score": data['momentum_score'].iloc[i]
                            }
                        else:
                            fvg = {
                                "type": "bearish",
                                "upper": c1['high'],
                                "lower": c3['low'],
                                "created_idx": i,
                                "touched": False,
                                "confluence_score": score,
                                "atr": data['atr'].iloc[i],
                                "volume_ratio": data['volume'].iloc[i] / data['volume_sma'].iloc[i],
                                "momentum_score": data['momentum_score'].iloc[i]
                            }
                        fvg_list.append(fvg)
                        print(f"DEBUG: Added FVG at {i} to list")
        
        print(f"DEBUG: Total FVGs after filtering: {len(fvg_list)}")
        return fvg_list
    
    def calculate_balanced_confluence_score(self, data, idx, direction):
        """Calculate balanced confluence score"""
        score = 0
        bar = data.iloc[idx]
        
        # RSI momentum confluence (more permissive)
        if direction == 'bullish':
            if bar['rsi'] < self.rsi_overbought:
                score += 1
            if bar['rsi'] < 50:  # Prefer lower RSI for bullish
                score += 1
        elif direction == 'bearish':
            if bar['rsi'] > self.rsi_oversold:
                score += 1
            if bar['rsi'] > 50:  # Prefer higher RSI for bearish
                score += 1
        
        # EMA trend confluence
        if direction == 'bullish' and bar['ema_fast'] >= bar['ema_slow']:
            score += 2
        elif direction == 'bearish' and bar['ema_fast'] <= bar['ema_slow']:
            score += 2
        
        # ADX trend strength (more lenient)
        if bar['adx'] > self.trend_strength_threshold:
            score += 1
        
        # Volume confirmation
        volume_ratio = bar['volume'] / bar['volume_sma']
        if volume_ratio > self.volume_multiplier:
            score += 1
        
        # Market structure alignment
        if direction == 'bullish' and bar['market_structure'] >= 0:
            score += 1
        elif direction == 'bearish' and bar['market_structure'] <= 0:
            score += 1
        
        # Momentum score
        if bar['momentum_score'] >= 3:
            score += 1
            
        return score
    
    def validate_fvg_context(self, data, idx, direction):
        """Validate FVG context for better entries"""
        # Check FVG size relative to ATR (reasonable size)
        if direction == 'bullish':
            fvg_size = data.iloc[idx-2]['low'] - data.iloc[idx]['high']
        else:
            fvg_size = data.iloc[idx-2]['high'] - data.iloc[idx]['low']
            
        atr_val = data['atr'].iloc[idx]
        
        # FVG should be reasonable size (more permissive)
        if fvg_size < atr_val * 0.1 or fvg_size > atr_val * 3.0:
            print(f"DEBUG: FVG size validation failed - size: {fvg_size}, ATR: {atr_val}")
            return False
        
        # Simplified context validation - just check if it's not in a tight range
        recent_bars = data.iloc[max(0, idx-3):idx]
        if len(recent_bars) < 2:
            return True  # Default to true if not enough data
        
        price_range = recent_bars['high'].max() - recent_bars['low'].min()
        avg_price = recent_bars['close'].mean()
        
        # If market is too choppy (very tight range), skip
        if price_range / avg_price < 0.005:  # Less than 0.5% range
            print(f"DEBUG: Market too choppy - range: {price_range/avg_price:.4f}")
            return False
        
        return True  # Much more permissive
    
    def calculate_dynamic_targets(self, entry_price, direction, atr_value):
        """Calculate conservative dynamic targets for higher win rate"""
        if direction == 'long':
            stop_loss = entry_price - (atr_value * self.atr_stop_multiplier)
            take_profit = entry_price + (atr_value * self.atr_tp_multiplier)
        else:
            stop_loss = entry_price + (atr_value * self.atr_stop_multiplier)
            take_profit = entry_price - (atr_value * self.atr_tp_multiplier)
            
        # Ensure minimum risk-reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if reward / risk < self.min_risk_reward:
            if direction == 'long':
                take_profit = entry_price + (risk * self.min_risk_reward)
            else:
                take_profit = entry_price - (risk * self.min_risk_reward)
                
        return stop_loss, take_profit
    
    def run_backtest(self, data: pd.DataFrame):
        """Optimized backtest targeting 80% win rate"""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex (timestamp)!")
        
        # Add indicators
        enhanced_data = self.add_indicators(data)
        
        # Detect smart money concepts
        order_blocks = detect_order_blocks(enhanced_data)
        liquidity_sweeps = detect_liquidity_sweeps(enhanced_data)
        
        # Detect quality FVGs
        fvg_list = self.detect_quality_fvg(enhanced_data)
        
        trades = []
        
        # Entry logic with balanced validation
        for fvg in fvg_list:
            idx_after_fvg = fvg["created_idx"] + 1
            if idx_after_fvg >= len(enhanced_data):
                continue
                
            # Reasonable search window
            max_search_bars = 15
            end_idx = min(idx_after_fvg + max_search_bars, len(enhanced_data))
                
            for i in range(idx_after_fvg, end_idx):
                bar = enhanced_data.iloc[i]
                if fvg["touched"]:
                    break
                
                # Skip ranging markets at entry
                if enhanced_data['market_regime'].iloc[i] == 0:
                    continue
                
                # FVG touch validation
                if fvg["type"] == "bullish" and bar["low"] <= fvg["upper"]:
                    if self.validate_balanced_entry(enhanced_data, i, 'long', order_blocks, liquidity_sweeps, fvg):
                        entry_price = min(bar["close"], fvg["upper"])
                        stop_loss, take_profit = self.calculate_dynamic_targets(
                            entry_price, 'long', fvg["atr"]
                        )
                        
                        trades.append({
                            "time": enhanced_data.index[i],
                            "type": "long",
                            "reason": "Optimized_FVG_touch",
                            "bar_index": i,
                            "confluence_score": fvg["confluence_score"],
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "atr": fvg["atr"],
                            "volume_ratio": fvg["volume_ratio"],
                            "momentum_score": fvg["momentum_score"],
                            "market_regime": enhanced_data['market_regime'].iloc[i]
                        })
                        fvg["touched"] = True
                        
                elif fvg["type"] == "bearish" and bar["high"] >= fvg["lower"]:
                    if self.validate_balanced_entry(enhanced_data, i, 'short', order_blocks, liquidity_sweeps, fvg):
                        entry_price = max(bar["close"], fvg["lower"])
                        stop_loss, take_profit = self.calculate_dynamic_targets(
                            entry_price, 'short', fvg["atr"]
                        )
                        
                        trades.append({
                            "time": enhanced_data.index[i],
                            "type": "short",
                            "reason": "Optimized_FVG_touch",
                            "bar_index": i,
                            "confluence_score": fvg["confluence_score"],
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "atr": fvg["atr"],
                            "volume_ratio": fvg["volume_ratio"],
                            "momentum_score": fvg["momentum_score"],
                            "market_regime": enhanced_data['market_regime'].iloc[i]
                        })
                        fvg["touched"] = True
        
        print(f"DEBUG: Optimized FVGs detected: {len(fvg_list)}, Trades generated: {len(trades)}")
        print(f"DEBUG: Order blocks detected: {len(order_blocks)}, Liquidity sweeps: {len(liquidity_sweeps)}")
        
        # Print statistics
        if fvg_list:
            avg_confluence = sum(fvg["confluence_score"] for fvg in fvg_list) / len(fvg_list)
            print(f"DEBUG: Average confluence score: {avg_confluence:.2f}")
        
        if trades:
            avg_momentum = sum(t["momentum_score"] for t in trades) / len(trades)
            print(f"DEBUG: Average momentum score: {avg_momentum:.2f}")
        
        return trades
    
    def validate_balanced_entry(self, data, idx, direction, order_blocks, liquidity_sweeps, fvg):
        """Balanced entry validation for optimal win rate"""
        bar = data.iloc[idx]
        
        # Basic market conditions
        current_rsi = bar['rsi']
        volume_ratio = bar['volume'] / bar['volume_sma']
        
        # Minimum volume requirement
        if volume_ratio < 0.8:
            return False
        
        # Direction-specific validations
        if direction == 'long':
            # For longs: avoid overbought conditions
            if current_rsi > 65:
                return False
                
            # Prefer bullish momentum
            if bar['momentum_score'] < 2:
                return False
                
            # Check for bullish market structure
            if bar['ema_fast'] < bar['ema_slow'] * 0.995:  # Allow small tolerance
                return False
                
        elif direction == 'short':
            # For shorts: avoid oversold conditions
            if current_rsi < 35:
                return False
                
            # Prefer bearish momentum
            if bar['momentum_score'] < 2:
                return False
                
            # Check for bearish market structure
            if bar['ema_fast'] > bar['ema_slow'] * 1.005:  # Allow small tolerance
                return False
        
        # Check confluence score at entry
        if fvg["confluence_score"] < self.min_confluence_score:
            return False
        
        # Supportive smart money concepts
        recent_sweeps = [s for s in liquidity_sweeps if abs(s['index'] - idx) <= 5]
        if recent_sweeps:
            return True
            
        nearby_obs = [ob for ob in order_blocks 
                     if ob['type'] == ('bullish' if direction == 'long' else 'bearish') 
                     and abs(ob['index'] - idx) <= 8]
        if nearby_obs:
            return True
        
        # Default: allow if basic conditions met
        return True