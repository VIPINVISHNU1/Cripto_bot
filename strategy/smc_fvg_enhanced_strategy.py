import pandas as pd
import numpy as np

class SMCFVGEnhancedStrategy:
    """
    Enhanced SMC FVG Strategy with improved signal filtering and dynamic risk management
    """
    def __init__(self, config, broker=None):
        self.config = config
        self.broker = broker
        
        # Enhanced parameters
        self.rsi_period = config.get("rsi_period", 14)
        self.ema_period = config.get("ema_period", 50)
        self.atr_period = config.get("atr_period", 14)
        self.volume_period = config.get("volume_period", 20)
        self.imbalance_threshold = config.get("imbalance_threshold", 0.0002)
        
        # Signal filtering parameters
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.volume_multiplier = config.get("volume_multiplier", 1.2)  # Above average volume
        self.atr_multiplier_sl = config.get("atr_multiplier_sl", 2.0)  # Stop loss ATR multiplier
        self.atr_multiplier_tp = config.get("atr_multiplier_tp", 3.0)  # Take profit ATR multiplier
        
    def calculate_indicators(self, data):
        """Calculate technical indicators"""
        df = data.copy()
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # EMA calculation
        df['ema'] = df['close'].ewm(span=self.ema_period).mean()
        
        # ATR calculation
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()
        
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(window=self.volume_period).mean()
        
        return df
    
    def detect_fvg(self, data, idx):
        """Enhanced FVG detection with validation"""
        if idx < 2:
            return None
            
        c1 = data.iloc[idx-2]
        c2 = data.iloc[idx-1]
        c3 = data.iloc[idx]
        
        # Bullish FVG: gap between c1.low and c3.high
        if c1['low'] > c3['high'] + (self.imbalance_threshold * c3['close']):
            return {
                "type": "bullish",
                "upper": c3['high'],
                "lower": c1['low'],
                "created_idx": idx,
                "touched": False,
                "strength": (c1['low'] - c3['high']) / c3['close']  # Gap size relative to price
            }
        
        # Bearish FVG: gap between c1.high and c3.low
        elif c1['high'] < c3['low'] - (self.imbalance_threshold * c3['close']):
            return {
                "type": "bearish",
                "upper": c1['high'],
                "lower": c3['low'],
                "created_idx": idx,
                "touched": False,
                "strength": (c3['low'] - c1['high']) / c3['close']  # Gap size relative to price
            }
        
        return None
    
    def validate_signal(self, data, idx, fvg_type):
        """Enhanced signal validation with confluence factors"""
        if idx >= len(data):
            return False, {}
            
        bar = data.iloc[idx]
        
        # Get indicator values
        rsi = bar.get('rsi', 50)
        ema = bar.get('ema', bar['close'])
        volume = bar.get('volume', 0)
        volume_ma = bar.get('volume_ma', volume)
        atr = bar.get('atr', 0)
        
        validation_score = 0
        reasons = []
        
        if fvg_type == "bullish":
            # RSI not oversold (avoid catching falling knife)
            if rsi > self.rsi_oversold:
                validation_score += 1
                reasons.append("RSI_favorable")
            
            # Price above EMA (trend confirmation)
            if bar['close'] > ema:
                validation_score += 2  # Higher weight for trend
                reasons.append("above_EMA")
                
        elif fvg_type == "bearish":
            # RSI not overbought
            if rsi < self.rsi_overbought:
                validation_score += 1
                reasons.append("RSI_favorable")
            
            # Price below EMA (trend confirmation)
            if bar['close'] < ema:
                validation_score += 2  # Higher weight for trend
                reasons.append("below_EMA")
        
        # Volume confirmation (require above average volume)
        if volume > volume_ma * self.volume_multiplier:
            validation_score += 1
            reasons.append("volume_confirm")
        
        # ATR validation (avoid low volatility periods)
        if atr > 0:
            atr_pct = atr / bar['close'] * 100
            if atr_pct > 0.5:  # At least 0.5% ATR
                validation_score += 1
                reasons.append("volatility_ok")
        
        # Require minimum score for signal validation
        min_score = 3  # Require at least 3 points for validation
        is_valid = validation_score >= min_score
        
        return is_valid, {
            "score": validation_score,
            "reasons": reasons,
            "rsi": rsi,
            "ema": ema,
            "volume_ratio": volume / volume_ma if volume_ma > 0 else 0,
            "atr_pct": atr / bar['close'] * 100 if atr > 0 else 0
        }
    
    def get_dynamic_levels(self, data, idx, entry_price, trade_type):
        """Calculate dynamic stop loss and take profit using ATR"""
        if idx >= len(data):
            return None, None
            
        bar = data.iloc[idx]
        atr = bar.get('atr', 0)
        
        if atr <= 0:
            # Fallback to fixed percentages if ATR not available
            if trade_type == "long":
                stop_price = entry_price * 0.99  # 1% stop loss
                tp_price = entry_price * 1.02    # 2% take profit
            else:
                stop_price = entry_price * 1.01  # 1% stop loss
                tp_price = entry_price * 0.98    # 2% take profit
        else:
            # ATR-based dynamic levels
            if trade_type == "long":
                stop_price = entry_price - (atr * self.atr_multiplier_sl)
                tp_price = entry_price + (atr * self.atr_multiplier_tp)
            else:
                stop_price = entry_price + (atr * self.atr_multiplier_sl)
                tp_price = entry_price - (atr * self.atr_multiplier_tp)
        
        return stop_price, tp_price
    
    def run_backtest(self, data: pd.DataFrame):
        """Enhanced backtest with technical indicators and filtering"""
        # Ensure data index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex (timestamp)!")
        
        # Calculate technical indicators
        data_with_indicators = self.calculate_indicators(data)
        
        fvg_list = []
        trades = []
        
        # FVG detection phase
        for i in range(2, len(data_with_indicators)):
            fvg = self.detect_fvg(data_with_indicators, i)
            if fvg:
                fvg_list.append(fvg)
        
        # Signal generation with validation
        for fvg in fvg_list:
            idx_after_fvg = fvg["created_idx"] + 1
            if idx_after_fvg >= len(data_with_indicators):
                continue
                
            for i in range(idx_after_fvg, len(data_with_indicators)):
                bar = data_with_indicators.iloc[i]
                if fvg["touched"]:
                    break
                
                # Check for FVG touch
                if fvg["type"] == "bullish" and bar["low"] <= fvg["upper"]:
                    # Validate signal before entry
                    is_valid, validation_info = self.validate_signal(data_with_indicators, i, "bullish")
                    
                    if is_valid:
                        # Calculate dynamic levels
                        entry_price = bar["close"]
                        stop_price, tp_price = self.get_dynamic_levels(data_with_indicators, i, entry_price, "long")
                        
                        trades.append({
                            "time": data_with_indicators.index[i],
                            "type": "long",
                            "reason": "FVG_bullish_validated",
                            "bar_index": i,
                            "entry_price": entry_price,
                            "stop_price": stop_price,
                            "tp_price": tp_price,
                            "validation": validation_info,
                            "fvg_strength": fvg["strength"]
                        })
                        fvg["touched"] = True
                        
                elif fvg["type"] == "bearish" and bar["high"] >= fvg["lower"]:
                    # Validate signal before entry
                    is_valid, validation_info = self.validate_signal(data_with_indicators, i, "bearish")
                    
                    if is_valid:
                        # Calculate dynamic levels
                        entry_price = bar["close"]
                        stop_price, tp_price = self.get_dynamic_levels(data_with_indicators, i, entry_price, "short")
                        
                        trades.append({
                            "time": data_with_indicators.index[i],
                            "type": "short",
                            "reason": "FVG_bearish_validated",
                            "bar_index": i,
                            "entry_price": entry_price,
                            "stop_price": stop_price,
                            "tp_price": tp_price,
                            "validation": validation_info,
                            "fvg_strength": fvg["strength"]
                        })
                        fvg["touched"] = True
        
        print(f"ENHANCED: FVGs detected: {len(fvg_list)}, Validated trades: {len(trades)}")
        return trades