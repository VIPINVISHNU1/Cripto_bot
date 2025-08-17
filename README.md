# Enhanced Crypto Trading Bot with Robust Backtesting

## Overview

This repository contains an enhanced cryptocurrency trading bot with comprehensive backtesting capabilities. The system implements Smart Money Concepts (SMC) trading strategies with sophisticated risk management and walk-forward optimization.

## Key Features

### 🎯 **Risk Management**
- **3% Risk Per Trade**: Position sizing based on percentage of current account balance
- **Dynamic Stop Loss/Take Profit**: Automatically optimized based on market conditions
- **Volatility-Adjusted Levels**: Stop loss and take profit levels adjust based on recent market volatility
- **Maximum Drawdown Controls**: Built-in safeguards against excessive losses

### 📊 **Comprehensive Backtesting**
- **Monthly Analysis**: Individual performance analysis for every month from August 2020 to August 2025
- **Walk-Forward Optimization**: Parameters optimized using historical data to avoid overfitting
- **Multiple Backtesting Modes**:
  - Standard backtesting with in-sample/out-of-sample split
  - Enhanced monthly backtesting with dynamic parameter optimization
  - Robust walk-forward backtesting with advanced overfitting protection

### 📈 **Strategy Implementation**
- **SMC FVG Strategy**: Fair Value Gap (Imbalance) detection and trading
- **Order Block Recognition**: Smart money institutional level identification
- **Liquidity Zone Analysis**: Key support/resistance level detection
- **Multiple Timeframe Support**: 1h, 4h, 1d timeframes supported

### 🔧 **Modular Configuration**
- **Hyperparameter Tuning**: Easy adjustment of all strategy parameters
- **Risk Parameter Configuration**: Customizable risk levels and targets
- **Flexible Time Periods**: Configure any backtesting period
- **Multiple Configuration Files**: Different setups for various scenarios

## Installation and Setup

### Prerequisites
```bash
pip install pandas numpy matplotlib pyyaml
```

### Optional (for live trading)
```bash
pip install python-binance
```

## Usage

### Configuration Files

The system supports multiple configuration files for different scenarios:

1. **config.yaml** - Default enhanced backtesting (full 5-year period)
2. **config_conservative.yaml** - Conservative parameters with walk-forward optimization
3. **config_test.yaml** - Quick testing with shorter periods

### Running Backtests

#### Enhanced Monthly Backtesting (Default)
```bash
python main.py
```

#### Robust Walk-Forward Backtesting
```bash
python main.py config_conservative.yaml
```

#### Quick Test (6 months)
```bash
python main.py config_test.yaml
```

## Configuration Parameters

### Core Strategy Settings
```yaml
strategy:
  symbol: "BTCUSDT"
  timeframe: "4h"
  position_size: 0.001  # For fixed sizing (deprecated)

risk:
  risk_per_trade_pct: 0.02  # 2% of balance per trade
  max_daily_loss: 100
  max_trades_per_day: 5

# Enhanced Parameters
target_monthly_return: 0.15  # 15% monthly target
stop_loss_range: [0.015, 0.02, 0.025, 0.03]
take_profit_range: [0.02, 0.025, 0.03, 0.035]
```

### Backtesting Modes
```yaml
enhanced_backtest: true              # Enable monthly analysis
walk_forward_optimization: true      # Enable robust optimization
lookback_months: 6                   # Historical data for optimization
```

## Output Files and Reports

### Generated Reports
- **Monthly Results Summary** (`data/monthly_results_summary.csv`)
- **Comprehensive Strategy Report** (`data/comprehensive_strategy_report.txt`)
- **Robust Analysis Report** (`data/robust_strategy_report.txt`)

### Visualizations
- **Strategy Performance Charts** (`data/strategy_performance_charts.png`)
- **Robust Strategy Analysis** (`data/robust_strategy_analysis.png`)

### Individual Trade Data
- **Monthly Trade Files** (`data/monthly_results/trades_YYYY-MM.csv`)
- **Robust Results** (`data/robust_results/`)

## Key Metrics Reported

### Performance Metrics
- **Net Profit (% and INR)**: Monthly and cumulative returns
- **Win Rate**: Percentage of profitable trades
- **Number of Trades**: Trading frequency analysis
- **Maximum Drawdown**: Risk assessment
- **Sharpe Ratio**: Risk-adjusted returns
- **Profit Factor**: Gross profit to gross loss ratio

### Robustness Analysis
- **Monthly Return Volatility**: Consistency measurement
- **Performance Degradation**: Overfitting detection
- **Target Achievement Rate**: Percentage of months above target
- **Consistency Score**: Overall stability rating

## Strategy Assessment Framework

The system provides automated strategy assessment based on:

### ✅ **Excellent Performance Indicators**
- Target achievement rate ≥ 60%
- Profitable months ≥ 75%
- Consistency score ≥ 80%
- Sharpe ratio ≥ 1.5
- No evidence of overfitting

### ⚠️ **Warning Indicators**
- Moderate target achievement (25-60%)
- Acceptable profitability (50-75%)
- Some return volatility
- Performance degradation detected

### ❌ **Poor Performance Indicators**
- Target achievement < 25%
- Profitable months < 50%
- High return volatility
- Strong overfitting evidence

## Example Results

Based on the comprehensive 61-month analysis (Aug 2020 - Aug 2025):

### Enhanced Backtesting Results
- **82% profitable months** (50 out of 61)
- **56.5% average monthly return**
- **60.7% months above 20% target**
- **2,694 total trades analyzed**
- **0.937 average Sharpe ratio**

### Assessment: 
- ✅ Excellent target achievement
- ✅ Robust profitability 
- ⚠️ High volatility detected
- ❌ Potential overfitting risk

## Risk Warnings

### Important Considerations
1. **Simulated Data**: Results based on synthetic historical data
2. **Market Conditions**: Past performance doesn't guarantee future results
3. **Overfitting Risk**: High volatility suggests parameter optimization issues
4. **Transaction Costs**: Real trading involves additional costs and slippage
5. **Market Impact**: Actual execution may differ from backtested results

## Recommended Usage

### For Strategy Development
1. Use **config_test.yaml** for quick parameter testing
2. Use **config_conservative.yaml** for robust evaluation
3. Compare results across different market periods
4. Focus on consistency over maximum returns

### For Parameter Optimization
1. Adjust `stop_loss_range` and `take_profit_range` arrays
2. Modify `risk_per_trade_pct` for different risk levels
3. Change `target_monthly_return` for different goals
4. Experiment with `lookback_months` for optimization period

## Advanced Features

### Walk-Forward Optimization
- Uses last N months to optimize parameters for current month
- Prevents look-ahead bias and reduces overfitting
- Provides more realistic performance estimates

### Volatility-Adjusted Levels
- Stop loss and take profit levels adjust based on recent volatility
- Helps maintain consistent risk across different market conditions
- Improves strategy robustness

### Multiple Broker Support
- MockBroker for backtesting (synthetic data)
- BinanceBroker for live trading (requires API keys)
- Easy integration of additional brokers

## Future Enhancements

### Planned Features
- [ ] Multi-timeframe analysis
- [ ] Portfolio-based backtesting
- [ ] Machine learning parameter optimization
- [ ] Real-time market data integration
- [ ] Advanced risk metrics (VaR, CVaR)
- [ ] Multiple strategy comparison framework

## Contributing

This system is designed to be modular and extensible. Key areas for contribution:
- Additional trading strategies
- Enhanced risk management features
- Performance optimization
- Additional data sources
- Improved visualization

## Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss and is not suitable for all investors. The authors assume no responsibility for any trading losses that may occur from using this software.

Always conduct thorough testing and validation before risking real capital, and never invest more than you can afford to lose.