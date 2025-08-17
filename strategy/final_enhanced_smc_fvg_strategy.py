import pandas as pd
import numpy as np
from utils.indicators import (
    calculate_rsi, calculate_ema, calculate_atr, calculate_adx,
    calculate_volume_sma, detect_market_structure, detect_order_blocks,
    detect_liquidity_sweeps
)

class FinalEnhancedSMCFVGStrategy:
    """Final Enhanced SMC FVG Strategy targeting 80% win rate with multiple improvements"""
    
    def __init__(self, config, broker=None):
        self.config = config
        self.broker = broker
        
        # Technical indicator periods
        self.rsi_period = 14
        self.ema_fast_period = 21
        self.ema_slow_period = 50
        self.atr_period = 14
        self.adx_period = 14  # Add missing attribute
        self.volume_period = 20
        
        # Multi-timeframe approach
        self.use_multi_timeframe = True
        
        # Balanced thresholds for 80% target
        self.min_confluence_score = 4     # Slightly lower for more trades
        self.min_trend_score = 2          # Reasonable trend requirement
        self.volume_threshold = 1.0       # Average volume
        
        # Very conservative risk management
        self.atr_stop_multiplier = 0.8   # Very tight stops
        self.atr_tp_multiplier = 4.0     # Large targets
        self.min_risk_reward = 3.0       # Excellent R:R
        
    def add_indicators(self, data):
        """Add comprehensive technical indicators"""
        data_copy = data.copy()
        
        # Core indicators
        data_copy['rsi'] = calculate_rsi(data_copy, self.rsi_period)
        data_copy['ema_fast'] = calculate_ema(data_copy, self.ema_fast_period)
        data_copy['ema_slow'] = calculate_ema(data_copy, self.ema_slow_period)
        data_copy['adx'] = calculate_adx(data_copy, self.adx_period)
        data_copy['atr'] = calculate_atr(data_copy, self.atr_period)
        data_copy['volume_sma'] = calculate_volume_sma(data_copy, self.volume_period)
        
        # Advanced indicators
        data_copy['trend_score'] = self.calculate_trend_score(data_copy)
        data_copy['momentum_quality'] = self.calculate_momentum_quality(data_copy)
        data_copy['market_context'] = self.calculate_market_context(data_copy)
        data_copy['volatility_regime'] = self.calculate_volatility_regime(data_copy)
        
        return data_copy
    
    def calculate_trend_score(self, data):
        """Calculate comprehensive trend score"""
        trend_score = np.zeros(len(data))
        
        for i in range(self.ema_slow_period, len(data)):
            score = 0
            
            # EMA slope analysis
            if i >= 5:
                ema_fast_slope = data['ema_fast'].iloc[i] - data['ema_fast'].iloc[i-5]
                ema_slow_slope = data['ema_slow'].iloc[i] - data['ema_slow'].iloc[i-5]
                
                if ema_fast_slope > 0 and ema_slow_slope > 0:
                    score += 2  # Both EMAs trending up
                elif ema_fast_slope < 0 and ema_slow_slope < 0:
                    score -= 2  # Both EMAs trending down
                elif ema_fast_slope > 0:
                    score += 1  # Fast EMA trending up
                elif ema_fast_slope < 0:
                    score -= 1  # Fast EMA trending down
            
            # EMA alignment
            if data['ema_fast'].iloc[i] > data['ema_slow'].iloc[i]:
                score += 1
            else:
                score -= 1
                
            # Price position relative to EMAs
            close = data['close'].iloc[i]
            ema_fast = data['ema_fast'].iloc[i]
            ema_slow = data['ema_slow'].iloc[i]
            
            if close > ema_fast > ema_slow:
                score += 2
            elif close < ema_fast < ema_slow:
                score -= 2
            elif close > ema_fast:
                score += 1
            elif close < ema_fast:
                score -= 1
                
            trend_score[i] = score
            
        return trend_score
    
    def calculate_momentum_quality(self, data):
        """Calculate momentum quality score"""
        momentum_quality = np.zeros(len(data))
        
        for i in range(20, len(data)):
            score = 0
            
            # RSI momentum
            rsi = data['rsi'].iloc[i]
            if 40 <= rsi <= 60:
                score += 2  # Neutral RSI
            elif 30 <= rsi <= 70:
                score += 1  # Reasonable RSI
            
            # Price momentum consistency
            recent_closes = data['close'].iloc[i-5:i+1]
            if len(recent_closes) >= 3:
                # Check for consistent direction
                ups = (recent_closes.diff() > 0).sum()
                downs = (recent_closes.diff() < 0).sum()
                
                if ups >= 4:
                    score += 2  # Strong up momentum
                elif downs >= 4:
                    score -= 2  # Strong down momentum
                elif ups >= 3:
                    score += 1  # Moderate up momentum
                elif downs >= 3:
                    score -= 1  # Moderate down momentum
            
            # Volume confirmation
            if i >= 3:
                avg_volume = data['volume'].iloc[i-3:i].mean()
                current_volume = data['volume'].iloc[i]
                if current_volume > avg_volume * 1.2:
                    score += 1
                    
            momentum_quality[i] = score
            
        return momentum_quality
    
    def calculate_market_context(self, data):
        """Calculate market context score"""
        context = np.zeros(len(data))
        
        for i in range(30, len(data)):
            score = 0
            
            # Volatility context
            current_atr = data['atr'].iloc[i]
            avg_atr = data['atr'].iloc[i-10:i].mean()
            
            if current_atr < avg_atr * 1.2:  # Not too volatile
                score += 1
            if current_atr > avg_atr * 0.8:  # Not too quiet
                score += 1
                
            # Range context
            recent_high = data['high'].iloc[i-10:i+1].max()
            recent_low = data['low'].iloc[i-10:i+1].min()
            current_price = data['close'].iloc[i]
            
            range_position = (current_price - recent_low) / (recent_high - recent_low)
            
            # Prefer entries not at extremes
            if 0.2 <= range_position <= 0.8:
                score += 2
            elif 0.1 <= range_position <= 0.9:
                score += 1
                
            context[i] = score
            
        return context
    
    def calculate_volatility_regime(self, data):
        """Determine volatility regime"""
        regime = np.zeros(len(data))
        
        for i in range(30, len(data)):
            current_atr = data['atr'].iloc[i]
            long_term_atr = data['atr'].iloc[i-30:i].mean()
            
            if current_atr > long_term_atr * 1.5:
                regime[i] = 2  # High volatility
            elif current_atr > long_term_atr * 1.2:
                regime[i] = 1  # Medium volatility
            else:
                regime[i] = 0  # Low volatility
                
        return regime

    def detect_high_probability_fvg(self, data):
        """Detect only highest probability FVGs"""
        fvg_list = []
        
        # More achievable search window
        for i in range(max(40, self.ema_slow_period + 10), len(data)):
            c1 = data.iloc[i-2]
            c2 = data.iloc[i-1]
            c3 = data.iloc[i]
            
            # Basic FVG detection
            bullish_fvg = c1['low'] > c3['high']
            bearish_fvg = c1['high'] < c3['low']
            
            if not (bullish_fvg or bearish_fvg):
                continue
                
            # Check if indicators are ready
            if pd.isna(data['rsi'].iloc[i]) or pd.isna(data['trend_score'].iloc[i]):
                continue
                
            direction = 'bullish' if bullish_fvg else 'bearish'
            
            # Calculate comprehensive score
            confluence_score = self.calculate_comprehensive_score(data, i, direction)
            
            if confluence_score >= self.min_confluence_score:
                # Final quality validation
                if self.validate_high_probability_setup(data, i, direction):
                    fvg_size = abs(c1['low'] - c3['high']) if bullish_fvg else abs(c1['high'] - c3['low'])
                    
                    fvg = {
                        "type": direction,
                        "upper": c3['high'] if direction == 'bullish' else c1['high'],
                        "lower": c1['low'] if direction == 'bullish' else c3['low'],
                        "created_idx": i,
                        "touched": False,
                        "confluence_score": confluence_score,
                        "atr": data['atr'].iloc[i],
                        "fvg_size": fvg_size,
                        "trend_score": data['trend_score'].iloc[i],
                        "momentum_quality": data['momentum_quality'].iloc[i],
                        "market_context": data['market_context'].iloc[i],
                        "volatility_regime": data['volatility_regime'].iloc[i]
                    }
                    fvg_list.append(fvg)
        
        return fvg_list
    
    def calculate_comprehensive_score(self, data, idx, direction):
        """Calculate comprehensive confluence score"""
        score = 0
        bar = data.iloc[idx]
        
        # Trend alignment (critical)
        trend_score = bar['trend_score']
        if direction == 'bullish' and trend_score >= self.min_trend_score:
            score += 3
        elif direction == 'bearish' and trend_score <= -self.min_trend_score:
            score += 3
        elif direction == 'bullish' and trend_score > 0:
            score += 1
        elif direction == 'bearish' and trend_score < 0:
            score += 1
        
        # Momentum quality
        momentum_quality = bar['momentum_quality']
        if direction == 'bullish' and momentum_quality >= 2:
            score += 2
        elif direction == 'bearish' and momentum_quality <= -2:
            score += 2
        elif direction == 'bullish' and momentum_quality > 0:
            score += 1
        elif direction == 'bearish' and momentum_quality < 0:
            score += 1
        
        # Market context
        if bar['market_context'] >= 3:
            score += 2
        elif bar['market_context'] >= 1:
            score += 1
        
        # RSI conditions
        rsi = bar['rsi']
        if direction == 'bullish':
            if 35 <= rsi <= 55:
                score += 2
            elif rsi <= 65:
                score += 1
        else:
            if 45 <= rsi <= 65:
                score += 2
            elif rsi >= 35:
                score += 1
        
        # Volume confirmation
        volume_ratio = bar['volume'] / bar['volume_sma']
        if volume_ratio > self.volume_threshold:
            score += 1
            if volume_ratio > 1.5:
                score += 1
        
        # ADX strength
        if bar['adx'] > 20:
            score += 1
            if bar['adx'] > 30:
                score += 1
        
        # Volatility regime
        if bar['volatility_regime'] == 1:  # Medium volatility is best
            score += 1
        
        return score
    
    def validate_high_probability_setup(self, data, idx, direction):
        """Final validation for high probability setups"""
        bar = data.iloc[idx]
        
        # FVG size validation
        fvg_size = abs(data.iloc[idx-2]['low'] - data.iloc[idx]['high']) if direction == 'bullish' else abs(data.iloc[idx-2]['high'] - data.iloc[idx]['low'])
        atr_val = bar['atr']
        
        # FVG should be reasonable size
        if fvg_size < atr_val * 0.2 or fvg_size > atr_val * 1.5:
            return False
        
        # Must have good trend alignment
        if abs(bar['trend_score']) < self.min_trend_score:
            return False
        
        # Momentum must be reasonable
        if abs(bar['momentum_quality']) < 1:
            return False
        
        # Avoid extreme RSI
        if bar['rsi'] > 75 or bar['rsi'] < 25:
            return False
        
        # Market context must be decent
        if bar['market_context'] < 1:
            return False
        
        return True
    
    def calculate_dynamic_targets(self, entry_price, direction, atr_value):
        """Calculate very conservative targets for maximum win rate"""
        if direction == 'long':
            stop_loss = entry_price - (atr_value * self.atr_stop_multiplier)
            take_profit = entry_price + (atr_value * self.atr_tp_multiplier)
        else:
            stop_loss = entry_price + (atr_value * self.atr_stop_multiplier)
            take_profit = entry_price - (atr_value * self.atr_tp_multiplier)
        
        # Ensure excellent risk-reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if reward / risk < self.min_risk_reward:
            if direction == 'long':
                take_profit = entry_price + (risk * self.min_risk_reward)
            else:
                take_profit = entry_price - (risk * self.min_risk_reward)
        
        return stop_loss, take_profit
    
    def run_backtest(self, data: pd.DataFrame):
        """Final enhanced backtest targeting 80%+ win rate"""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex (timestamp)!")
        
        # Add all indicators
        enhanced_data = self.add_indicators(data)
        
        # Detect smart money concepts
        order_blocks = detect_order_blocks(enhanced_data)
        liquidity_sweeps = detect_liquidity_sweeps(enhanced_data)
        
        # Detect only highest probability FVGs
        fvg_list = self.detect_high_probability_fvg(enhanced_data)
        
        trades = []
        
        # Ultra-selective entry logic
        for fvg in fvg_list:
            idx_after_fvg = fvg["created_idx"] + 1
            if idx_after_fvg >= len(enhanced_data):
                continue
                
            # Slightly longer window for more opportunities
            max_search_bars = 12
            end_idx = min(idx_after_fvg + max_search_bars, len(enhanced_data))
                
            for i in range(idx_after_fvg, end_idx):
                bar = enhanced_data.iloc[i]
                if fvg["touched"]:
                    break
                
                # FVG touch with final validation
                if fvg["type"] == "bullish" and bar["low"] <= fvg["upper"]:
                    if self.validate_final_entry(enhanced_data, i, 'long', fvg):
                        entry_price = min(bar["close"], fvg["upper"])
                        stop_loss, take_profit = self.calculate_dynamic_targets(
                            entry_price, 'long', fvg["atr"]
                        )
                        
                        trades.append({
                            "time": enhanced_data.index[i],
                            "type": "long",
                            "reason": "High_Probability_FVG",
                            "bar_index": i,
                            "confluence_score": fvg["confluence_score"],
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "risk_reward": abs(take_profit - entry_price) / abs(entry_price - stop_loss),
                            "trend_score": fvg["trend_score"],
                            "momentum_quality": fvg["momentum_quality"]
                        })
                        fvg["touched"] = True
                        
                elif fvg["type"] == "bearish" and bar["high"] >= fvg["lower"]:
                    if self.validate_final_entry(enhanced_data, i, 'short', fvg):
                        entry_price = max(bar["close"], fvg["lower"])
                        stop_loss, take_profit = self.calculate_dynamic_targets(
                            entry_price, 'short', fvg["atr"]
                        )
                        
                        trades.append({
                            "time": enhanced_data.index[i],
                            "type": "short",
                            "reason": "High_Probability_FVG",
                            "bar_index": i,
                            "confluence_score": fvg["confluence_score"],
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "risk_reward": abs(take_profit - entry_price) / abs(entry_price - stop_loss),
                            "trend_score": fvg["trend_score"],
                            "momentum_quality": fvg["momentum_quality"]
                        })
                        fvg["touched"] = True
        
        print(f"DEBUG: High Probability FVGs detected: {len(fvg_list)}, Trades generated: {len(trades)}")
        
        if fvg_list:
            avg_confluence = sum(fvg["confluence_score"] for fvg in fvg_list) / len(fvg_list)
            avg_trend = sum(fvg["trend_score"] for fvg in fvg_list) / len(fvg_list)
            print(f"DEBUG: Average confluence score: {avg_confluence:.2f}, Average trend score: {avg_trend:.2f}")
        
        if trades:
            avg_rr = sum(t["risk_reward"] for t in trades) / len(trades)
            print(f"DEBUG: Average risk-reward ratio: {avg_rr:.2f}")
        
        return trades
    
    def validate_final_entry(self, data, idx, direction, fvg):
        """Final entry validation"""
        bar = data.iloc[idx]
        
        # Must maintain high quality at entry
        if fvg["confluence_score"] < self.min_confluence_score:
            return False
        
        # Check real-time conditions
        current_rsi = bar['rsi']
        current_trend = bar['trend_score']
        current_momentum = bar['momentum_quality']
        
        if direction == 'long':
            # For long entries
            if current_rsi > 70:  # Avoid overbought
                return False
            if current_trend <= 0:  # Must have bullish trend
                return False
            if current_momentum < 0:  # Must have positive momentum
                return False
        else:
            # For short entries
            if current_rsi < 30:  # Avoid oversold
                return False
            if current_trend >= 0:  # Must have bearish trend
                return False
            if current_momentum > 0:  # Must have negative momentum
                return False
        
        # Volume confirmation
        volume_ratio = bar['volume'] / bar['volume_sma']
        if volume_ratio < 0.8:  # Minimum volume
            return False
        
        return True