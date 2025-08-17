# Parameter Optimization Guide

## Overview

This document explains how to use the parameter optimization system to tune your trading strategy for maximum risk-adjusted returns.

## System Architecture

The parameter optimization system consists of several key components:

### 1. ParameterOptimizer Class (`optimization/parameter_optimizer.py`)
- Systematic grid search across parameter combinations
- In-sample/out-of-sample validation to prevent overfitting
- Comprehensive performance metrics calculation
- Results storage and analysis

### 2. Scripts
- `optimize_parameters.py` - Full parameter optimization with real Binance data
- `test_optimization.py` - Demo version using local CSV data
- `debug_optimization.py` - Debug tool for troubleshooting

## Quick Start

### Demo Mode (Recommended for Testing)
```bash
python test_optimization.py
```
This uses the local CSV data and runs a quick optimization with reduced parameter grid.

### Full Optimization Mode
```bash
python optimize_parameters.py
```
This connects to Binance API and runs full parameter optimization.

## Parameter Ranges

The system optimizes the following key parameters:

### Risk Management Parameters
- **stop_loss_pct**: [0.5%, 1%, 1.5%, 2%, 2.5%] - Stop loss percentage
- **take_profit_pct**: [1%, 1.5%, 2%, 2.5%, 3%, 3.5%, 4%] - Take profit percentage  
- **position_size**: [0.0005, 0.001, 0.002, 0.003] - Risk per trade

### Strategy-Specific Parameters
- **imbalance_threshold**: [0.0001, 0.0002, 0.0003, 0.0005, 0.001] - FVG detection threshold

## Performance Metrics

For each parameter combination, the system calculates:

### Primary Metrics
- **Sharpe Ratio**: Risk-adjusted returns (higher is better)
- **Total P&L**: Absolute profit/loss
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of winning trades

### Additional Metrics
- **Number of Trades**: Trade frequency
- **Profit Factor**: Gross profit / gross loss
- **Average Win/Loss**: Mean profit per winning/losing trade
- **Volatility**: Annualized return standard deviation
- **Average Profit per Month**: Monthly profitability estimate

## Results Analysis

The system identifies the best parameter sets across five criteria:

1. **Best Out-of-Sample Sharpe Ratio**: Maximum risk-adjusted returns
2. **Best Out-of-Sample Total P&L**: Maximum absolute profit
3. **Best Out-of-Sample Profit Factor**: Best reward/risk ratio
4. **Most Consistent Performance**: Smallest difference between in/out-sample results
5. **Best Risk-Adjusted Performance**: Highest Sharpe/drawdown ratio

## Output Files

All results are saved in `data/optimization_results/`:

- `optimization_results_YYYYMMDD_HHMMSS.json` - Raw optimization results
- `analysis_YYYYMMDD_HHMMSS.json` - Best parameter sets analysis
- `detailed_results_YYYYMMDD_HHMMSS.csv` - Tabular results for Excel analysis
- `summary_report_YYYYMMDD_HHMMSS.txt` - Human-readable summary (if using optimize_parameters.py)

## Understanding Results

### Interpreting Sharpe Ratio
- **> 1.0**: Excellent performance
- **0.5 - 1.0**: Good performance  
- **0 - 0.5**: Acceptable performance
- **< 0**: Poor performance (losing strategy)

### Interpreting Consistency
The "Most Consistent" parameters show similar performance between in-sample and out-of-sample periods, indicating lower overfitting risk.

### Trade-offs
- Higher returns often come with higher volatility
- More trades may increase transaction costs
- Tighter stops reduce drawdown but may increase whipsaw losses

## Customizing Parameter Ranges

To modify the parameter grid, edit the `define_parameter_grid()` method in `ParameterOptimizer`:

```python
def define_parameter_grid(self) -> Dict[str, List]:
    return {
        'stop_loss_pct': [0.005, 0.01, 0.015],  # Your custom ranges
        'take_profit_pct': [0.015, 0.02, 0.025],
        'position_size': [0.001, 0.002],
        'imbalance_threshold': [0.0002, 0.0005]
    }
```

## Best Practices

### 1. Anti-Overfitting Measures
- Always validate on out-of-sample data
- Prefer consistent performers over peak performers
- Use sufficient historical data (multiple market cycles)
- Limit parameter combinations to avoid curve-fitting

### 2. Implementation Guidelines
- Start with "Most Consistent" parameters for live trading
- Monitor performance closely after implementation
- Re-optimize periodically as market conditions change
- Consider transaction costs in your position sizing

### 3. Statistical Requirements
- Minimum 20+ out-of-sample trades for statistical significance
- Test across different market conditions (trending, ranging, volatile)
- Consider seasonal effects and market regime changes

## Integration with Live Trading

Once you've identified optimal parameters, update your `config.yaml`:

```yaml
# Example optimized parameters
stop_loss_pct: 0.015        # 1.5%
take_profit_pct: 0.025      # 2.5%
position_size: 0.001        # 0.1% risk per trade
imbalance_threshold: 0.0002 # FVG detection sensitivity
```

## Troubleshooting

### Common Issues

1. **No valid results**: Increase parameter ranges or reduce minimum trade requirements
2. **All negative results**: Strategy may not be profitable with current data/parameters
3. **High variance between in/out-sample**: Reduce parameter grid to avoid overfitting

### Debug Tools

Use `debug_optimization.py` to test individual parameter combinations:
```bash
python debug_optimization.py
```

## Advanced Usage

### Custom Strategies
To optimize different strategies, replace the strategy import in the optimization scripts:
```python
from strategy.your_strategy import YourStrategy
strategy = YourStrategy(config["strategy"], broker)
```

### Additional Metrics
Add custom metrics by extending the `calculate_comprehensive_metrics()` method.

### Multi-Objective Optimization
The current system focuses on Sharpe ratio optimization, but can be extended for multi-objective optimization (e.g., maximize return while minimizing drawdown).

## Future Enhancements

- Bayesian optimization for more efficient parameter search
- Walk-forward optimization for time-varying parameters
- Monte Carlo simulation for robustness testing
- Multi-asset parameter optimization