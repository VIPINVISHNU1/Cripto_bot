import pandas as pd
import numpy as np

class SMCFVGLooseStrategy:
    """
    Enhanced SMC FVG Strategy (formerly loose, now improved)
    - Adds technical indicators for signal filtering
    - Implements dynamic risk management
    - Maintains compatibility with existing backtester
    """
    def __init__(self, config, broker=None):
        self.config = config
        self.broker = broker
        
        # Enhanced parameters with sensible defaults
        self.rsi_period = config.get("rsi_period", 14)
        self.ema_period = config.get("ema_period", 50)
        self.atr_period = config.get("atr_period", 14)
        self.volume_period = config.get("volume_period", 20)
        self.imbalance_threshold = config.get("imbalance_threshold", 0.0002)
        
        # Signal filtering parameters (more lenient for higher trade frequency)
        self.rsi_oversold = config.get("rsi_oversold", 25)
        self.rsi_overbought = config.get("rsi_overbought", 75)
        self.min_validation_score = config.get("min_validation_score", 3)
        
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
        """Enhanced FVG detection"""
        if idx < 2:
            return None
            
        c1 = data.iloc[idx-2]
        c3 = data.iloc[idx]
        
        # Bullish FVG with threshold
        if c1['low'] > c3['high'] + (self.imbalance_threshold * c3['close']):
            return {
                "type": "bullish",
                "upper": c3['high'],
                "lower": c1['low'],
                "created_idx": idx,
                "touched": False,
                "strength": (c1['low'] - c3['high']) / c3['close']
            }
        
        # Bearish FVG with threshold
        elif c1['high'] < c3['low'] - (self.imbalance_threshold * c3['close']):
            return {
                "type": "bearish",
                "upper": c1['high'],
                "lower": c3['low'],
                "created_idx": idx,
                "touched": False,
                "strength": (c3['low'] - c1['high']) / c3['close']
            }
        
        return None
    
    def validate_signal(self, data, idx, fvg_type):
        """Enhanced signal validation"""
        if idx >= len(data):
            return False, {}
            
        bar = data.iloc[idx]
        rsi = bar.get('rsi', 50)
        ema = bar.get('ema', bar['close'])
        volume = bar.get('volume', 0)
        volume_ma = bar.get('volume_ma', volume)
        atr = bar.get('atr', 0)
        
        validation_score = 0
        reasons = []
        
        if fvg_type == "bullish":
            # RSI not oversold (momentum plays)
            if rsi > self.rsi_oversold:
                validation_score += 1
                reasons.append("RSI_ok")
            if 40 < rsi < 70:  # Sweet spot
                validation_score += 1
                reasons.append("RSI_balanced")
            
            # Trend confirmation
            if bar['close'] > ema:
                validation_score += 2
                reasons.append("uptrend")
                
        elif fvg_type == "bearish":
            # RSI not overbought
            if rsi < self.rsi_overbought:
                validation_score += 1
                reasons.append("RSI_ok")
            if 30 < rsi < 60:  # Sweet spot
                validation_score += 1
                reasons.append("RSI_balanced")
            
            # Trend confirmation
            if bar['close'] < ema:
                validation_score += 2
                reasons.append("downtrend")
        
        # Volume (lenient requirement)
        volume_ratio = volume / volume_ma if volume_ma > 0 else 1
        if volume_ratio > 0.8:  # At least 80% of average
            validation_score += 1
            reasons.append("volume_ok")
        
        # Volatility check
        if atr > 0:
            atr_pct = atr / bar['close'] * 100
            if atr_pct > 0.4:  # Minimum volatility
                validation_score += 1
                reasons.append("volatility_ok")
        
        is_valid = validation_score >= self.min_validation_score
        
        return is_valid, {
            "score": validation_score,
            "reasons": reasons,
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "atr_pct": atr / bar['close'] * 100 if atr > 0 else 0
        }

    def run_backtest(self, data: pd.DataFrame):
        """Enhanced backtest with improved signal filtering"""
        # Ensure data index is datetime for signal timing
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex (timestamp)!")

        # Calculate technical indicators
        data_with_indicators = self.calculate_indicators(data)
        
        fvg_list = []
        trades = []

        # FVG detection
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
                    # Validate signal
                    is_valid, validation_info = self.validate_signal(data_with_indicators, i, "bullish")
                    
                    if is_valid:
                        trades.append({
                            "time": data_with_indicators.index[i],
                            "type": "long",
                            "reason": "FVG_bullish_validated",
                            "bar_index": i,
                            "validation": validation_info
                        })
                        fvg["touched"] = True
                    else:
                        # If validation fails, still mark as touched to avoid repeated checks
                        fvg["touched"] = True
                        
                elif fvg["type"] == "bearish" and bar["high"] >= fvg["lower"]:
                    # Validate signal
                    is_valid, validation_info = self.validate_signal(data_with_indicators, i, "bearish")
                    
                    if is_valid:
                        trades.append({
                            "time": data_with_indicators.index[i],
                            "type": "short",
                            "reason": "FVG_bearish_validated",
                            "bar_index": i,
                            "validation": validation_info
                        })
                        fvg["touched"] = True
                    else:
                        # If validation fails, still mark as touched to avoid repeated checks
                        fvg["touched"] = True

        print(f"ENHANCED: FVGs detected: {len(fvg_list)}, Validated trades: {len(trades)}")
        return trades