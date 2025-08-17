import yaml
from broker.mock_broker import MockBroker
from strategy.smc_fvg_loose_strategy import SMCFVGLooseStrategy

from utils.logger import get_logger
from utils.risk import RiskManager
from backtest.enhanced_backtester import EnhancedBacktester
from backtest.robust_backtester import RobustBacktester

import pandas as pd

# Try to import real broker, fallback to mock if not available
try:
    from broker.binance import BinanceBroker
    from backtest.backtester import Backtester
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("Binance client not available, using mock broker only")

def load_config(config_file="config.yaml"):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def main():
    # Allow configuration file selection
    import sys
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        print(f"Using configuration file: {config_file}")
    else:
        config_file = "config.yaml"
    
    config = load_config(config_file)
    logger = get_logger(config["logging"])
    risk_manager = RiskManager(config["risk"], logger)

    # Use mock broker for enhanced backtesting, real broker for regular backtesting
    if config.get("enhanced_backtest", False):
        broker = MockBroker(config["broker"], logger)
        logger.info("Using MockBroker for enhanced backtesting")
    else:
        if BINANCE_AVAILABLE:
            broker = BinanceBroker(config["broker"], logger)
            logger.info("Using BinanceBroker for regular backtesting")
        else:
            logger.warning("Binance client not available, falling back to MockBroker")
            broker = MockBroker(config["broker"], logger)

    strategy = SMCFVGLooseStrategy(config["strategy"], broker)

    if config["mode"] == "backtest":
        # Check which type of backtesting to use
        if config.get("walk_forward_optimization", False):
            logger.info("Starting robust walk-forward backtesting...")
            
            # Use robust backtester with walk-forward optimization
            robust_backtester = RobustBacktester(config, strategy, logger)
            monthly_results = robust_backtester.run_robust_backtest()
            
            logger.info(f"Robust backtest completed. Analyzed {len(monthly_results)} months.")
            
        elif config.get("enhanced_backtest", False):
            logger.info("Starting enhanced monthly backtesting...")
            
            # Use enhanced backtester for monthly analysis
            enhanced_backtester = EnhancedBacktester(config, strategy, logger)
            monthly_results = enhanced_backtester.run_monthly_backtest()
            
            logger.info(f"Enhanced backtest completed. Analyzed {len(monthly_results)} months.")
        else:
            if BINANCE_AVAILABLE:
                # Original backtesting approach
                orig_get_historical_klines = broker.get_historical_klines
                def get_historical_klines_with_index(symbol, timeframe, start, end):
                    df = orig_get_historical_klines(symbol, timeframe, start, end)
                    if "timestamp" in df.columns:
                        df = df.set_index("timestamp")
                    return df
                broker.get_historical_klines = get_historical_klines_with_index

                backtester = Backtester(config, strategy, logger)
                backtester.run()
            else:
                logger.error("Cannot run regular backtest without Binance client. Please enable enhanced_backtest or install python-binance.")
            
    elif config["mode"] == "live":
        if BINANCE_AVAILABLE:
            logger.info("Starting live trading mode.")
            strategy.run_live()
        else:
            logger.error("Cannot run live trading without Binance client.")
    else:
        logger.error("Invalid mode specified in config.yaml.")

if __name__ == "__main__":
    main()