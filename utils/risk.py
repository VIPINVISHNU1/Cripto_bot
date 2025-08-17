class RiskManager:
    def __init__(self, config, logger):
        self.max_daily_loss = config["max_daily_loss"]
        self.max_trades_per_day = config["max_trades_per_day"]
        self.logger = logger
        self.trades_today = 0
        self.daily_loss = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = config.get("max_consecutive_losses", 5)
        self.max_position_size = config.get("max_position_size", 1.0)
        self.min_position_size = config.get("min_position_size", 0.001)
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0

    def register_trade(self, pnl, equity=None):
        """Register a completed trade and update risk metrics."""
        self.trades_today += 1
        self.total_trades += 1
        self.total_pnl += pnl
        
        # Track wins/losses
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.daily_loss += abs(pnl)
            self.consecutive_losses += 1
        
        # Update drawdown if equity is provided
        if equity is not None:
            if equity > self.peak_equity:
                self.peak_equity = equity
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Check risk limits
        if self.daily_loss > self.max_daily_loss:
            self.logger.warning(f"Max daily loss exceeded! Current: {self.daily_loss:.2f}, Max: {self.max_daily_loss}")
            return False
            
        if self.trades_today > self.max_trades_per_day:
            self.logger.warning(f"Max trades per day exceeded! Current: {self.trades_today}, Max: {self.max_trades_per_day}")
            return False
            
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.warning(f"Max consecutive losses reached! Count: {self.consecutive_losses}")
            return False
            
        return True
    
    def validate_position_size(self, position_size):
        """Validate and adjust position size within risk limits."""
        return max(self.min_position_size, min(position_size, self.max_position_size))
    
    def get_risk_metrics(self):
        """Get current risk metrics."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        return {
            "total_trades": self.total_trades,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "max_drawdown": self.max_drawdown * 100,  # Convert to percentage
            "consecutive_losses": self.consecutive_losses,
            "daily_loss": self.daily_loss,
            "trades_today": self.trades_today
        }
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of each new day)."""
        self.trades_today = 0
        self.daily_loss = 0