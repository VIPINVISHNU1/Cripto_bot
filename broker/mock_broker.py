import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class MockBroker:
    """Mock broker for backtesting without needing real API access"""
    
    def __init__(self, config, logger):
        self.logger = logger
        self.api_key = config.get("api_key", "mock_key")
        self.api_secret = config.get("api_secret", "mock_secret")
        self.testnet = config.get("testnet", True)

    def get_historical_klines(self, symbol, interval, start_str, end_str=None):
        """
        Generate mock historical data for backtesting
        This replaces the need for real Binance API calls
        """
        try:
            start_date = pd.to_datetime(start_str)
            if end_str:
                end_date = pd.to_datetime(end_str)
            else:
                end_date = datetime.now()
            
            # Generate date range based on interval
            if interval == "4h":
                freq = "4H"
            elif interval == "1h":
                freq = "1H"
            elif interval == "1d":
                freq = "1D"
            else:
                freq = "4H"  # Default to 4h
            
            date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
            
            # Generate realistic price data
            base_price = self._get_base_price_for_symbol_date(symbol, start_date)
            
            # Set seed for reproducible data
            np.random.seed(hash(symbol + start_str) % 2**32)
            
            prices = []
            current_price = base_price
            
            for _ in range(len(date_range)):
                # Random walk with slight upward bias
                change = np.random.normal(0.001, 0.02)  # 0.1% average growth, 2% volatility
                current_price *= (1 + change)
                current_price = max(current_price, 1000)  # Price floor
                prices.append(current_price)
            
            # Generate OHLCV data
            data = []
            for i, price in enumerate(prices):
                if i == 0:
                    open_price = price
                else:
                    open_price = prices[i-1]
                
                close_price = price
                
                # Generate intrabar high/low
                volatility = 0.005  # 0.5% intrabar volatility
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility)))
                
                volume = np.random.uniform(1000, 5000)
                
                data.append({
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data, index=date_range)
            df.index.name = 'timestamp'
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating mock klines: {e}")
            return None

    def _get_base_price_for_symbol_date(self, symbol, date):
        """Get realistic base price based on symbol and date"""
        if "BTC" in symbol:
            year = date.year
            if year <= 2020:
                return np.random.uniform(8000, 15000)
            elif year == 2021:
                return np.random.uniform(15000, 50000)
            elif year == 2022:
                return np.random.uniform(15000, 40000)
            elif year == 2023:
                return np.random.uniform(20000, 45000)
            elif year == 2024:
                return np.random.uniform(30000, 70000)
            else:  # 2025+
                return np.random.uniform(60000, 120000)
        elif "ETH" in symbol:
            year = date.year
            if year <= 2020:
                return np.random.uniform(200, 800)
            elif year == 2021:
                return np.random.uniform(800, 4000)
            elif year == 2022:
                return np.random.uniform(800, 3000)
            elif year == 2023:
                return np.random.uniform(1000, 3000)
            elif year == 2024:
                return np.random.uniform(1500, 4000)
            else:  # 2025+
                return np.random.uniform(3000, 8000)
        else:
            # Default for other symbols
            return np.random.uniform(1, 100)