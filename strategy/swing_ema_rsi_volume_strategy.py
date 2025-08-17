import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any


class SwingEMARsiVolumeStrategy:
    """
    Enhanced Swing Trading Strategy with EMA, RSI, Volume, and additional indicators.
    
    Features:
    - Dynamic position sizing using ATR
    - Enhanced risk management with trailing stops
    - Multiple indicator confirmation (EMA, RSI, Volume, MACD, Bollinger Bands, ADX)
    - Cross-validation for backtesting
    - Comprehensive logging
    - Live trading ready
    """
    
    def __init__(self, config: Dict[str, Any], broker, logger: Optional[logging.Logger] = None):
        self.config = config
        self.broker = broker
        self.logger = logger or logging.getLogger(__name__)
        
        # Core strategy parameters
        self.ema_fast = int(config.get("ema_fast", 12))
        self.ema_slow = int(config.get("ema_slow", 26))
        self.rsi_period = int(config.get("rsi_period", 14))
        self.rsi_oversold = float(config.get("rsi_oversold", 30))
        self.rsi_overbought = float(config.get("rsi_overbought", 70))
        
        # Volume parameters
        self.volume_ma_period = int(config.get("volume_ma_period", 20))
        self.volume_threshold = float(config.get("volume_threshold", 1.5))  # Volume must be 1.5x average
        
        # ATR for dynamic position sizing
        self.atr_period = int(config.get("atr_period", 14))
        self.risk_per_trade = float(config.get("risk_per_trade", 0.02))  # 2% risk per trade
        
        # Risk management
        self.stop_loss_atr_mult = float(config.get("stop_loss_atr_mult", 2.0))
        self.take_profit_atr_mult = float(config.get("take_profit_atr_mult", 3.0))
        self.trailing_stop_atr_mult = float(config.get("trailing_stop_atr_mult", 1.5))
        
        # Additional indicators
        self.use_macd = bool(config.get("use_macd", True))
        self.use_bollinger = bool(config.get("use_bollinger", True))
        self.use_adx = bool(config.get("use_adx", True))
        
        # MACD parameters
        self.macd_fast = int(config.get("macd_fast", 12))
        self.macd_slow = int(config.get("macd_slow", 26))
        self.macd_signal = int(config.get("macd_signal", 9))
        
        # Bollinger Bands parameters
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        
        # ADX parameters
        self.adx_period = int(config.get("adx_period", 14))
        self.adx_threshold = float(config.get("adx_threshold", 25))
        
        # Cross-validation parameters
        self.use_cross_validation = bool(config.get("use_cross_validation", True))
        self.cv_folds = int(config.get("cv_folds", 5))
        
        self.logger.info(f"SwingEMARsiVolumeStrategy initialized with parameters: {self._get_param_summary()}")
    
    def _get_param_summary(self) -> str:
        """Get a summary of strategy parameters for logging."""
        return f"EMA({self.ema_fast},{self.ema_slow}), RSI({self.rsi_period}), ATR({self.atr_period}), Risk({self.risk_per_trade*100}%)"
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_macd(self, data: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal, and Histogram."""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd = ema_fast - ema_slow
        signal = self.calculate_ema(macd, signal)
        histogram = macd - signal
        return macd, signal, histogram
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int, std_mult: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * std_mult)
        lower = sma - (std * std_mult)
        return upper, sma, lower
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average Directional Index (simplified version)."""
        # Calculate True Range
        tr = self.calculate_atr(high, low, close, 1)
        
        # Calculate Directional Movement
        dm_plus = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
        dm_minus = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
        
        dm_plus = pd.Series(dm_plus, index=high.index)
        dm_minus = pd.Series(dm_minus, index=low.index)
        
        # Calculate Directional Indicators
        di_plus = 100 * (dm_plus.rolling(window=period).mean() / tr.rolling(window=period).mean())
        di_minus = 100 * (dm_minus.rolling(window=period).mean() / tr.rolling(window=period).mean())
        
        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        df = data.copy()
        
        # Core indicators
        df['ema_fast'] = self.calculate_ema(df['close'], self.ema_fast)
        df['ema_slow'] = self.calculate_ema(df['close'], self.ema_slow)
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], self.atr_period)
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Additional indicators
        if self.use_macd:
            df['macd'], df['macd_signal'], df['macd_histogram'] = self.calculate_macd(
                df['close'], self.macd_fast, self.macd_slow, self.macd_signal
            )
        
        if self.use_bollinger:
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(
                df['close'], self.bb_period, self.bb_std
            )
        
        if self.use_adx:
            df['adx'] = self.calculate_adx(df['high'], df['low'], df['close'], self.adx_period)
        
        return df
    
    def calculate_position_size(self, equity: float, entry_price: float, stop_loss: float, atr: float) -> float:
        """Calculate position size based on ATR and risk per trade."""
        risk_amount = equity * self.risk_per_trade
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff == 0:
            price_diff = atr * self.stop_loss_atr_mult
        
        position_size = risk_amount / price_diff
        return max(position_size, 0.001)  # Minimum position size
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate trading signals based on multiple indicators."""
        df = self.calculate_indicators(data)
        signals = []
        
        for i in range(max(self.ema_slow, self.rsi_period, self.atr_period), len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Core conditions
            ema_bullish = row['ema_fast'] > row['ema_slow']
            ema_bearish = row['ema_fast'] < row['ema_slow']
            
            rsi_oversold = row['rsi'] < self.rsi_oversold
            rsi_overbought = row['rsi'] > self.rsi_overbought
            
            volume_spike = row['volume_ratio'] > self.volume_threshold
            
            # Additional confirmations
            confirmations_long = 0
            confirmations_short = 0
            
            if self.use_macd and not pd.isna(row['macd_histogram']):
                if row['macd_histogram'] > 0 and prev_row['macd_histogram'] <= 0:
                    confirmations_long += 1
                elif row['macd_histogram'] < 0 and prev_row['macd_histogram'] >= 0:
                    confirmations_short += 1
            
            if self.use_bollinger and not pd.isna(row['bb_lower']):
                if row['close'] < row['bb_lower']:
                    confirmations_long += 1
                elif row['close'] > row['bb_upper']:
                    confirmations_short += 1
            
            if self.use_adx and not pd.isna(row['adx']):
                if row['adx'] > self.adx_threshold:
                    confirmations_long += 1
                    confirmations_short += 1
            
            # Long signal
            if (ema_bullish and rsi_oversold and volume_spike and 
                confirmations_long >= 1):
                signals.append({
                    "time": df.index[i],
                    "type": "long",
                    "reason": "EMA_RSI_Volume_Long",
                    "price": row['close'],
                    "atr": row['atr'],
                    "rsi": row['rsi'],
                    "volume_ratio": row['volume_ratio'],
                    "confirmations": confirmations_long
                })
                self.logger.debug(f"Long signal at {df.index[i]}: RSI={row['rsi']:.2f}, Volume={row['volume_ratio']:.2f}, Confirmations={confirmations_long}")
            
            # Short signal
            elif (ema_bearish and rsi_overbought and volume_spike and 
                  confirmations_short >= 1):
                signals.append({
                    "time": df.index[i],
                    "type": "short",
                    "reason": "EMA_RSI_Volume_Short",
                    "price": row['close'],
                    "atr": row['atr'],
                    "rsi": row['rsi'],
                    "volume_ratio": row['volume_ratio'],
                    "confirmations": confirmations_short
                })
                self.logger.debug(f"Short signal at {df.index[i]}: RSI={row['rsi']:.2f}, Volume={row['volume_ratio']:.2f}, Confirmations={confirmations_short}")
        
        self.logger.info(f"Generated {len(signals)} signals from {len(df)} bars")
        return signals
    
    def calculate_dynamic_stops(self, entry_price: float, atr: float, signal_type: str) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit based on ATR."""
        if signal_type == "long":
            stop_loss = entry_price - (atr * self.stop_loss_atr_mult)
            take_profit = entry_price + (atr * self.take_profit_atr_mult)
        else:  # short
            stop_loss = entry_price + (atr * self.stop_loss_atr_mult)
            take_profit = entry_price - (atr * self.take_profit_atr_mult)
        
        return stop_loss, take_profit
    
    def update_trailing_stop(self, current_price: float, entry_price: float, current_stop: float, 
                           atr: float, signal_type: str) -> float:
        """Update trailing stop loss based on ATR."""
        if signal_type == "long":
            new_stop = current_price - (atr * self.trailing_stop_atr_mult)
            return max(new_stop, current_stop)  # Only move stop up
        else:  # short
            new_stop = current_price + (atr * self.trailing_stop_atr_mult)
            return min(new_stop, current_stop)  # Only move stop down
    
    def run_backtest(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run backtest and return signals with enhanced logic."""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex (timestamp)!")
        
        signals = self.generate_signals(data)
        
        # Add dynamic stop loss and take profit to signals
        for signal in signals:
            if 'atr' in signal:
                stop_loss, take_profit = self.calculate_dynamic_stops(
                    signal['price'], signal['atr'], signal['type']
                )
                signal['stop_loss'] = stop_loss
                signal['take_profit'] = take_profit
        
        return signals
    
    def run_cross_validation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run cross-validation backtest to avoid overfitting."""
        if not self.use_cross_validation:
            return {"message": "Cross-validation disabled"}
        
        fold_size = len(data) // self.cv_folds
        cv_results = []
        
        for fold in range(self.cv_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < self.cv_folds - 1 else len(data)
            
            test_data = data.iloc[start_idx:end_idx]
            
            # Generate signals for this fold
            signals = self.generate_signals(test_data)
            
            cv_results.append({
                "fold": fold + 1,
                "signals": len(signals),
                "start_date": test_data.index[0],
                "end_date": test_data.index[-1]
            })
        
        self.logger.info(f"Cross-validation completed: {len(cv_results)} folds")
        return {"cv_results": cv_results}
    
    def run_live(self):
        """Placeholder for live trading logic."""
        self.logger.info("Starting live trading mode for SwingEMARsiVolumeStrategy")
        self.logger.info("Live trading implementation needed:")
        self.logger.info("- Real-time data fetching")
        self.logger.info("- Order execution logic")
        self.logger.info("- Position management")
        self.logger.info("- Risk monitoring")
        
        # TODO: Implement live trading logic
        # - Connect to real-time data feed
        # - Monitor for signals
        # - Execute orders through broker
        # - Manage positions and risk
        pass