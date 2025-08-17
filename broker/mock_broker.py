import pandas as pd
import os

class MockBroker:
    """Mock broker for backtesting that uses local CSV data"""
    
    def __init__(self, config, logger):
        self.logger = logger
        self.config = config
        
    def get_historical_klines(self, symbol, interval, start_str, end_str=None):
        """Load data from local CSV file"""
        try:
            # Use the existing CSV file
            csv_path = "BTCUSDT_4h_ohlcv.csv"
            if os.path.exists(csv_path):
                data = pd.read_csv(csv_path)
                data["timestamp"] = pd.to_datetime(data["timestamp"])
                data.set_index("timestamp", inplace=True)
                
                # Filter by date range if specified
                if start_str:
                    start_date = pd.to_datetime(start_str)
                    data = data[data.index >= start_date]
                
                if end_str:
                    end_date = pd.to_datetime(end_str)
                    data = data[data.index <= end_date]
                
                return data[["open", "high", "low", "close", "volume"]].astype(float)
            else:
                self.logger.error(f"CSV file {csv_path} not found")
                return None
        except Exception as e:
            self.logger.error(f"Error loading data from CSV: {e}")
            return None