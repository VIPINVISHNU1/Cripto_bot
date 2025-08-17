class RiskManager:
    def __init__(self, config, logger):
        self.max_daily_loss = config["max_daily_loss"]
        self.max_trades_per_day = config["max_trades_per_day"]
        self.risk_per_trade_pct = config.get("risk_per_trade_pct", 0.03)  # 3% default
        self.logger = logger
        self.trades_today = 0
        self.daily_loss = 0

    def calculate_position_size(self, balance, entry_price, stop_loss_price):
        """Calculate position size based on risk percentage of balance"""
        if stop_loss_price == 0 or entry_price == 0:
            return 0
        
        risk_amount = balance * self.risk_per_trade_pct
        price_diff = abs(entry_price - stop_loss_price)
        
        if price_diff == 0:
            return 0
            
        position_size = risk_amount / price_diff
        return position_size

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