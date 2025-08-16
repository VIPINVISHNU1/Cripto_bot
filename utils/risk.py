class RiskManager:
    def __init__(self, config, logger):
        self.max_daily_loss = config["max_daily_loss"]
        self.max_trades_per_day = config["max_trades_per_day"]
        self.logger = logger
        self.trades_today = 0
        self.daily_loss = 0

    def register_trade(self, pnl):
        self.trades_today += 1
        self.daily_loss += max(0, -pnl)
        if self.daily_loss > self.max_daily_loss:
            self.logger.warning("Max daily loss exceeded! Stopping trading.")
            return False
        if self.trades_today > self.max_trades_per_day:
            self.logger.warning("Max trades per day exceeded! Stopping trading.")
            return False
        return True