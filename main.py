import yaml
from broker.binance import BinanceBroker
from broker.mock_broker import MockBroker
from strategy.smc_imbalance_orderblock import SMCImbalanceOrderBlockStrategy
from strategy.smc_orderblock import SMCOrderBlockStrategy
from strategy.smc_fvg_bounce_strategy import SMCFVGStrategy
from strategy.smc_fvg_loose_strategy import SMCFVGLooseStrategy
from strategy.optimized_smc_fvg_strategy import OptimizedSMCFVGStrategy

from utils.logger import get_logger
from utils.risk import RiskManager
from backtest.backtester import Backtester

import pandas as pd

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    logger = get_logger(config["logging"])
    risk_manager = RiskManager(config["risk"], logger)

    # Use mock broker for backtesting to avoid network issues
    broker = MockBroker(config["broker"], logger)

    # Use optimized strategy targeting 80% win rate
    strategy = OptimizedSMCFVGStrategy(config["strategy"], broker)

    if config["mode"] == "backtest":
        # Ensure all data uses timestamp as index!
        # Patch broker's get_historical_klines to always set_index
        orig_get_historical_klines = broker.get_historical_klines
        def get_historical_klines_with_index(symbol, timeframe, start, end):
            df = orig_get_historical_klines(symbol, timeframe, start, end)
            if df is not None and "timestamp" in df.columns:
                df = df.set_index("timestamp")
            return df
        broker.get_historical_klines = get_historical_klines_with_index

        backtester = Backtester(config, strategy, logger)
        backtester.run()
    elif config["mode"] == "live":
        logger.info("Starting live trading mode.")
        strategy.run_live()
    else:
        logger.error("Invalid mode specified in config.yaml.")

if __name__ == "__main__":
    main()