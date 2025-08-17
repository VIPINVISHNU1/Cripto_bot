import pandas as pd
import numpy as np
from utils.indicators import (
    calculate_rsi, calculate_ema, calculate_atr, calculate_adx,
    calculate_volume_sma, detect_market_structure, detect_order_blocks,
    detect_liquidity_sweeps
)

class HighWinRateSMCFVGStrategy:
    """High Win Rate SMC FVG Strategy targeting 80%+ win rate"""
    
    def __init__(self, config, broker=None):
        self.config = config
        self.broker = broker
        
        # Conservative parameters for high win rate
        self.rsi_period = 14
        self.ema_fast_period = 21
        self.ema_slow_period = 50
        self.atr_period = 14
        self.adx_period = 14
        self.volume_period = 20
        
        # More achievable parameters for balanced performance
        self.min_confluence_score = 4      # Achievable threshold
        self.min_momentum_score = 2        # Reasonable momentum requirement
        self.volume_multiplier = 1.0       # Average volume
        self.trend_strength_threshold = 15 # Moderate trend requirement
        
        # Conservative risk management
        self.atr_stop_multiplier = 1.0     # Very tight stops
        self.atr_tp_multiplier = 4.0       # Large targets for good R:R
        self.min_risk_reward = 3.0         # Excellent risk-reward
        
    def add_indicators(self, data):
        """Add all technical indicators"""
        data_copy = data.copy()
        
        # Core indicators
        data_copy['rsi'] = calculate_rsi(data_copy, self.rsi_period)
        data_copy['ema_fast'] = calculate_ema(data_copy, self.ema_fast_period)
        data_copy['ema_slow'] = calculate_ema(data_copy, self.ema_slow_period)
        data_copy['adx'] = calculate_adx(data_copy, self.adx_period)
        data_copy['atr'] = calculate_atr(data_copy, self.atr_period)
        data_copy['volume_sma'] = calculate_volume_sma(data_copy, self.volume_period)
        
        # Market structure and momentum
        data_copy['market_structure'] = detect_market_structure(data_copy)
        data_copy['momentum_score'] = self.calculate_momentum_score(data_copy)
        data_copy['trend_alignment'] = self.calculate_trend_alignment(data_copy)
        
        return data_copy
    
    def calculate_momentum_score(self, data):
        """Calculate comprehensive momentum score"""
        momentum_score = np.zeros(len(data))
        
        for i in range(10, len(data)):
            score = 0
            
            # Multi-timeframe momentum
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                score += 1
            if data['close'].iloc[i] > data['close'].iloc[i-3]:
                score += 1
            if data['close'].iloc[i] > data['close'].iloc[i-5]:
                score += 1
            if data['close'].iloc[i] > data['close'].iloc[i-10]:
                score += 1
                
            # EMA momentum
            if data['ema_fast'].iloc[i] > data['ema_fast'].iloc[i-1]:
                score += 1
                
            momentum_score[i] = score
            
        return momentum_score
    
    def calculate_trend_alignment(self, data):
        """Calculate trend alignment score"""
        alignment = np.zeros(len(data))
        
        for i in range(self.ema_slow_period, len(data)):
            score = 0
            
            # EMA alignment
            if data['ema_fast'].iloc[i] > data['ema_slow'].iloc[i]:
                score += 2  # Bullish alignment
            else:
                score -= 2  # Bearish alignment
                
            # Price vs EMAs
            close = data['close'].iloc[i]
            ema_fast = data['ema_fast'].iloc[i]
            ema_slow = data['ema_slow'].iloc[i]
            
            if close > ema_fast > ema_slow:
                score += 2  # Strong bullish
            elif close < ema_fast < ema_slow:
                score -= 2  # Strong bearish
            elif close > ema_fast:
                score += 1  # Mild bullish
            elif close < ema_fast:
                score -= 1  # Mild bearish
                
            alignment[i] = score
            
        return alignment

    def detect_premium_fvg(self, data):
        """Detect only premium quality FVGs"""
        fvg_list = []
        
        for i in range(max(30, self.ema_slow_period + 5), len(data)):
            c1 = data.iloc[i-2]
            c2 = data.iloc[i-1]
            c3 = data.iloc[i]
            
            # Basic FVG detection
            bullish_fvg = c1['low'] > c3['high']
            bearish_fvg = c1['high'] < c3['low']
            
            if not (bullish_fvg or bearish_fvg):
                continue
                
            direction = 'bullish' if bullish_fvg else 'bearish'
            
            # Calculate comprehensive score
            confluence_score = self.calculate_premium_confluence_score(data, i, direction)
            
            if confluence_score >= self.min_confluence_score:
                # Additional premium filters
                if self.validate_premium_setup(data, i, direction):
                    fvg = {
                        "type": direction,
                        "upper": c3['high'] if direction == 'bullish' else c1['high'],
                        "lower": c1['low'] if direction == 'bullish' else c3['low'],
                        "created_idx": i,
                        "touched": False,
                        "confluence_score": confluence_score,
                        "atr": data['atr'].iloc[i],
                        "volume_ratio": data['volume'].iloc[i] / data['volume_sma'].iloc[i],
                        "momentum_score": data['momentum_score'].iloc[i],
                        "trend_alignment": data['trend_alignment'].iloc[i]
                    }
                    fvg_list.append(fvg)
        
        return fvg_list
    
    def calculate_premium_confluence_score(self, data, idx, direction):
        """Calculate premium confluence score with strict criteria"""
        score = 0
        bar = data.iloc[idx]
        
        # RSI conditions (strict)
        rsi = bar['rsi']
        if direction == 'bullish':
            if 30 <= rsi <= 50:  # Oversold but not extreme
                score += 2
            elif rsi <= 60:  # Not overbought
                score += 1
        else:
            if 50 <= rsi <= 70:  # Overbought but not extreme
                score += 2
            elif rsi >= 40:  # Not oversold
                score += 1
        
        # Trend alignment (critical)
        trend_align = bar['trend_alignment']
        if direction == 'bullish' and trend_align >= 3:
            score += 3
        elif direction == 'bearish' and trend_align <= -3:
            score += 3
        elif direction == 'bullish' and trend_align > 0:
            score += 1
        elif direction == 'bearish' and trend_align < 0:
            score += 1
        
        # ADX trend strength
        if bar['adx'] > self.trend_strength_threshold:
            score += 2
            if bar['adx'] > 30:
                score += 1
        
        # Volume confirmation
        volume_ratio = bar['volume'] / bar['volume_sma']
        if volume_ratio > self.volume_multiplier:
            score += 1
            if volume_ratio > 1.5:
                score += 1
        
        # Momentum confirmation
        if bar['momentum_score'] >= self.min_momentum_score:
            score += 2
            if bar['momentum_score'] >= 4:
                score += 1
        
        # Market structure
        ms = bar['market_structure']
        if direction == 'bullish' and ms > 0:
            score += 1
        elif direction == 'bearish' and ms < 0:
            score += 1
            
        return score
    
    def validate_premium_setup(self, data, idx, direction):
        """Validate premium setup quality"""
        # FVG size validation
        if direction == 'bullish':
            fvg_size = data.iloc[idx-2]['low'] - data.iloc[idx]['high']
        else:
            fvg_size = data.iloc[idx-2]['high'] - data.iloc[idx]['low']
            
        atr_val = data['atr'].iloc[idx]
        
        # FVG should be meaningful but not too large
        if fvg_size < atr_val * 0.15 or fvg_size > atr_val * 2.0:
            return False
        
        # Check for clean setup - no conflicting signals nearby
        lookback = data.iloc[max(0, idx-5):idx]
        
        if direction == 'bullish':
            # For bullish: look for overall bullish context
            recent_trend = (lookback['close'].iloc[-1] >= lookback['close'].iloc[0])
            ema_support = (data['close'].iloc[idx] >= data['ema_fast'].iloc[idx])
            return recent_trend and ema_support
        else:
            # For bearish: look for overall bearish context
            recent_trend = (lookback['close'].iloc[-1] <= lookback['close'].iloc[0])
            ema_resistance = (data['close'].iloc[idx] <= data['ema_fast'].iloc[idx])
            return recent_trend and ema_resistance
    
    def calculate_dynamic_targets(self, entry_price, direction, atr_value):
        """Calculate conservative targets for high win rate"""
        if direction == 'long':
            stop_loss = entry_price - (atr_value * self.atr_stop_multiplier)
            take_profit = entry_price + (atr_value * self.atr_tp_multiplier)
        else:
            stop_loss = entry_price + (atr_value * self.atr_stop_multiplier)
            take_profit = entry_price - (atr_value * self.atr_tp_multiplier)
        
        # Ensure excellent risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if reward / risk < self.min_risk_reward:
            if direction == 'long':
                take_profit = entry_price + (risk * self.min_risk_reward)
            else:
                take_profit = entry_price - (risk * self.min_risk_reward)
        
        return stop_loss, take_profit
    
    def run_backtest(self, data: pd.DataFrame):
        """High win rate backtest"""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex (timestamp)!")
        
        # Add indicators
        enhanced_data = self.add_indicators(data)
        
        # Detect smart money concepts
        order_blocks = detect_order_blocks(enhanced_data)
        liquidity_sweeps = detect_liquidity_sweeps(enhanced_data)
        
        # Detect premium FVGs only
        fvg_list = self.detect_premium_fvg(enhanced_data)
        
        trades = []
        
        # Very selective entry logic
        for fvg in fvg_list:
            idx_after_fvg = fvg["created_idx"] + 1
            if idx_after_fvg >= len(enhanced_data):
                continue
                
            # Short search window for fresh signals
            max_search_bars = 10
            end_idx = min(idx_after_fvg + max_search_bars, len(enhanced_data))
                
            for i in range(idx_after_fvg, end_idx):
                bar = enhanced_data.iloc[i]
                if fvg["touched"]:
                    break
                
                # FVG touch with final validation
                if fvg["type"] == "bullish" and bar["low"] <= fvg["upper"]:
                    if self.validate_premium_entry(enhanced_data, i, 'long', order_blocks, liquidity_sweeps, fvg):
                        entry_price = min(bar["close"], fvg["upper"])
                        stop_loss, take_profit = self.calculate_dynamic_targets(
                            entry_price, 'long', fvg["atr"]
                        )
                        
                        trades.append({
                            "time": enhanced_data.index[i],
                            "type": "long",
                            "reason": "Premium_FVG_touch",
                            "bar_index": i,
                            "confluence_score": fvg["confluence_score"],
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "atr": fvg["atr"],
                            "risk_reward": abs(take_profit - entry_price) / abs(entry_price - stop_loss)
                        })
                        fvg["touched"] = True
                        
                elif fvg["type"] == "bearish" and bar["high"] >= fvg["lower"]:
                    if self.validate_premium_entry(enhanced_data, i, 'short', order_blocks, liquidity_sweeps, fvg):
                        entry_price = max(bar["close"], fvg["lower"])
                        stop_loss, take_profit = self.calculate_dynamic_targets(
                            entry_price, 'short', fvg["atr"]
                        )
                        
                        trades.append({
                            "time": enhanced_data.index[i],
                            "type": "short",
                            "reason": "Premium_FVG_touch",
                            "bar_index": i,
                            "confluence_score": fvg["confluence_score"],
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "atr": fvg["atr"],
                            "risk_reward": abs(take_profit - entry_price) / abs(entry_price - stop_loss)
                        })
                        fvg["touched"] = True
        
        print(f"DEBUG: Premium FVGs detected: {len(fvg_list)}, Trades generated: {len(trades)}")
        
        if fvg_list:
            avg_confluence = sum(fvg["confluence_score"] for fvg in fvg_list) / len(fvg_list)
            print(f"DEBUG: Average confluence score: {avg_confluence:.2f}")
        
        if trades:
            avg_rr = sum(t["risk_reward"] for t in trades) / len(trades)
            print(f"DEBUG: Average risk-reward ratio: {avg_rr:.2f}")
        
        return trades
    
    def validate_premium_entry(self, data, idx, direction, order_blocks, liquidity_sweeps, fvg):
        """Final premium entry validation"""
        bar = data.iloc[idx]
        
        # Must have excellent confluence at entry
        if fvg["confluence_score"] < self.min_confluence_score:
            return False
        
        # Volume must be decent
        volume_ratio = bar['volume'] / bar['volume_sma']
        if volume_ratio < 0.9:
            return False
        
        # Momentum must be aligned
        if bar['momentum_score'] < self.min_momentum_score:
            return False
        
        # Trend must be reasonable (not too strict)
        if direction == 'long':
            if bar['trend_alignment'] < 1:  # Lowered from 2
                return False
            if bar['rsi'] > 70:  # Still avoid extreme overbought
                return False
        else:
            if bar['trend_alignment'] > -1:  # Lowered from -2
                return False
            if bar['rsi'] < 30:  # Still avoid extreme oversold
                return False
        
        # Check for supportive smart money activity
        recent_sweeps = [s for s in liquidity_sweeps if abs(s['index'] - idx) <= 3]
        nearby_obs = [ob for ob in order_blocks 
                     if ob['type'] == ('bullish' if direction == 'long' else 'bearish') 
                     and abs(ob['index'] - idx) <= 5]
        
        # Either supportive concepts or excellent technical setup
        smart_money_support = len(recent_sweeps) > 0 or len(nearby_obs) > 0
        excellent_technical = (fvg["confluence_score"] >= 8 and 
                             bar['momentum_score'] >= 4)
        
        return smart_money_support or excellent_technical