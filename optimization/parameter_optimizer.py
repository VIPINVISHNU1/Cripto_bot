import pandas as pd
import numpy as np
import itertools
import json
import os
from datetime import datetime
import copy
from typing import Dict, List, Tuple, Any

class ParameterOptimizer:
    """
    Systematic parameter optimization for trading strategies.
    Performs grid search while avoiding overfitting through out-of-sample validation.
    """
    
    def __init__(self, config, strategy, broker, logger):
        self.config = config
        self.strategy = strategy
        self.broker = broker
        self.logger = logger
        self.results = []
        
        # Create results directory
        self.results_dir = "data/optimization_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def define_parameter_grid(self) -> Dict[str, List]:
        """
        Define parameter ranges for optimization.
        Returns dictionary with parameter names and their test values.
        """
        return {
            'stop_loss_pct': [0.005, 0.01, 0.015, 0.02, 0.025],  # 0.5% to 2.5%
            'take_profit_pct': [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04],  # 1% to 4%
            'position_size': [0.0005, 0.001, 0.002, 0.003],  # Risk per trade
            'imbalance_threshold': [0.0001, 0.0002, 0.0003, 0.0005, 0.001]  # FVG threshold
        }
    
    def generate_parameter_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """
        Generate all combinations of parameters for grid search.
        """
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = []
        
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            # Add constraint: take_profit should be >= stop_loss
            if param_dict['take_profit_pct'] >= param_dict['stop_loss_pct']:
                combinations.append(param_dict)
        
        self.logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    def run_single_backtest(self, params: Dict, data: pd.DataFrame, tag: str) -> Dict:
        """
        Run backtest with specific parameter set.
        """
        from backtest.backtester import Backtester
        
        # Create modified config with new parameters
        modified_config = copy.deepcopy(self.config)
        modified_config.update(params)
        
        # Update strategy config if it has imbalance_threshold
        if hasattr(self.strategy, 'imbalance_threshold') and 'imbalance_threshold' in params:
            self.strategy.imbalance_threshold = params['imbalance_threshold']
        
        # Create backtester with modified config
        backtester = Backtester(modified_config, self.strategy, self.logger)
        
        # Override specific parameters
        backtester.stop_loss_pct = params.get('stop_loss_pct', backtester.stop_loss_pct)
        backtester.take_profit_pct = params.get('take_profit_pct', backtester.take_profit_pct)
        backtester.position_size = params.get('position_size', backtester.position_size)
        
        # Run simulation
        results = backtester.simulate_trades(data, tag)
        
        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(results, params)
        return metrics
    
    def calculate_comprehensive_metrics(self, backtest_results: Dict, params: Dict) -> Dict:
        """
        Calculate comprehensive performance metrics for parameter evaluation.
        """
        trades = backtest_results["trades"]
        eq_curve = backtest_results["equity_curve"]
        final_balance = backtest_results["final_balance"]
        
        if not trades or not eq_curve:
            return {
                'params': params,
                'total_pnl': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_profit_per_month': 0,
                'volatility': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'final_balance': self.config.get('initial_balance', 10000)
            }
        
        try:
            # Basic metrics
            pnl_list = [t["pnl"] for t in trades]
            total_pnl = sum(pnl_list)
            num_trades = len(trades)
            
            # Win/Loss analysis
            wins = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p <= 0]
            win_rate = len(wins) / num_trades if num_trades > 0 else 0
            
            avg_win = float(np.mean(wins)) if wins else 0
            avg_loss = float(np.mean(losses)) if losses else 0
            largest_win = float(max(wins)) if wins else 0
            largest_loss = float(min(losses)) if losses else 0
            
            # Profit factor
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 1e-8
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Risk metrics
            max_dd = self._max_drawdown(eq_curve)
            sharpe = self._sharpe_ratio(eq_curve)
            
            # Estimate monthly metrics (assuming data covers full period)
            days_in_period = len(eq_curve) * 4/24  # 4h bars to days
            months_in_period = days_in_period / 30.44  # average days per month
            avg_profit_per_month = total_pnl / months_in_period if months_in_period > 0 else 0
            
            # Volatility (annualized return std)
            if len(eq_curve) > 1:
                eq_array = np.array(eq_curve)
                returns = np.diff(eq_array) / eq_array[:-1]
                volatility = float(np.std(returns) * np.sqrt(365 * 6))  # 4h bars annualized
            else:
                volatility = 0
            
            return {
                'params': params,
                'total_pnl': float(total_pnl),
                'num_trades': int(num_trades),
                'win_rate': float(win_rate),
                'avg_profit_per_month': float(avg_profit_per_month),
                'volatility': volatility,
                'max_drawdown': float(max_dd),
                'sharpe_ratio': float(sharpe),
                'profit_factor': float(profit_factor),
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'final_balance': float(final_balance)
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {
                'params': params,
                'total_pnl': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_profit_per_month': 0,
                'volatility': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'final_balance': self.config.get('initial_balance', 10000)
            }
    
    def optimize_parameters(self, max_combinations: int = None) -> List[Dict]:
        """
        Run parameter optimization across full parameter grid.
        """
        self.logger.info("Starting parameter optimization...")
        
        # Get data
        data = self.broker.get_historical_klines(
            self.config["strategy"]["symbol"],
            self.config["strategy"]["timeframe"],
            self.config["backtest"]["start"],
            self.config["backtest"]["end"]
        )
        
        if data is None or data.empty:
            raise ValueError("No data available for optimization")
        
        # Split into in-sample and out-of-sample
        train_pct = self.config.get("in_sample_pct", 0.7)
        split_idx = int(len(data) * train_pct)
        data_in_sample = data.iloc[:split_idx]
        data_out_sample = data.iloc[split_idx:]
        
        self.logger.info(f"Data split: {len(data_in_sample)} in-sample, {len(data_out_sample)} out-of-sample bars")
        
        # Generate parameter combinations
        param_grid = self.define_parameter_grid()
        param_combinations = self.generate_parameter_combinations(param_grid)
        
        # Limit combinations if specified
        if max_combinations and len(param_combinations) > max_combinations:
            param_combinations = param_combinations[:max_combinations]
            self.logger.info(f"Limited to {max_combinations} combinations")
        
        results = []
        
        # Run optimization
        for i, params in enumerate(param_combinations):
            if i % 10 == 0:
                self.logger.info(f"Processing combination {i+1}/{len(param_combinations)}")
            
            try:
                # In-sample results
                in_sample_metrics = self.run_single_backtest(params, data_in_sample, f"opt_in_{i}")
                
                # Out-of-sample results  
                out_sample_metrics = self.run_single_backtest(params, data_out_sample, f"opt_out_{i}")
                
                # Combine results
                combined_result = {
                    'combination_id': i,
                    'params': params,
                    'in_sample': in_sample_metrics,
                    'out_sample': out_sample_metrics,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(combined_result)
                
            except Exception as e:
                self.logger.error(f"Error in combination {i}: {e}")
                continue
        
        self.results = results
        self.save_results()
        
        self.logger.info(f"Optimization completed. {len(results)} successful combinations.")
        return results
    
    def save_results(self):
        """Save optimization results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/optimization_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filename}")
    
    def analyze_results(self) -> Dict:
        """
        Analyze optimization results and find best parameter sets.
        """
        if not self.results:
            self.logger.warning("No results to analyze")
            return {}
        
        # Convert to DataFrame for easier analysis
        analysis_data = []
        for result in self.results:
            row = {
                'combination_id': result['combination_id'],
                **result['params'],
                'in_sharpe': result['in_sample']['sharpe_ratio'],
                'out_sharpe': result['out_sample']['sharpe_ratio'],
                'in_total_pnl': result['in_sample']['total_pnl'],
                'out_total_pnl': result['out_sample']['total_pnl'],
                'in_max_dd': result['in_sample']['max_drawdown'],
                'out_max_dd': result['out_sample']['max_drawdown'],
                'in_win_rate': result['in_sample']['win_rate'],
                'out_win_rate': result['out_sample']['win_rate'],
                'in_trades': result['in_sample']['num_trades'],
                'out_trades': result['out_sample']['num_trades'],
                'in_volatility': result['in_sample']['volatility'],
                'out_volatility': result['out_sample']['volatility'],
                'in_profit_factor': result['in_sample']['profit_factor'],
                'out_profit_factor': result['out_sample']['profit_factor']
            }
            analysis_data.append(row)
        
        df = pd.DataFrame(analysis_data)
        
        # Filter for reasonable results (positive out-of-sample trades)
        valid_results = df[df['out_trades'] > 5]  # At least 5 trades out-of-sample
        
        if valid_results.empty:
            self.logger.warning("No valid results found (require >5 out-of-sample trades)")
            return {}
        
        # Find best parameter sets by different criteria
        best_results = {
            'best_out_sharpe': self._get_best_by_metric(valid_results, 'out_sharpe'),
            'best_out_pnl': self._get_best_by_metric(valid_results, 'out_total_pnl'),
            'best_out_profit_factor': self._get_best_by_metric(valid_results, 'out_profit_factor'),
            'most_consistent': self._get_most_consistent(valid_results),
            'best_risk_adjusted': self._get_best_risk_adjusted(valid_results)
        }
        
        # Save analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = f"{self.results_dir}/analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(best_results, f, indent=2, default=str)
        
        # Save detailed results CSV
        csv_file = f"{self.results_dir}/detailed_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Analysis saved to {analysis_file} and {csv_file}")
        return best_results
    
    def _get_best_by_metric(self, df: pd.DataFrame, metric: str) -> Dict:
        """Get best result by specific metric."""
        best_row = df.loc[df[metric].idxmax()]
        return best_row.to_dict()
    
    def _get_most_consistent(self, df: pd.DataFrame) -> Dict:
        """Find most consistent performer (smallest difference between in/out sample)."""
        df = df.copy()
        df['sharpe_diff'] = abs(df['in_sharpe'] - df['out_sharpe'])
        df['pnl_diff'] = abs(df['in_total_pnl'] - df['out_total_pnl'])
        
        # Consistency score (lower is better)
        df['consistency_score'] = df['sharpe_diff'] + (df['pnl_diff'] / df['in_total_pnl'].abs())
        
        most_consistent = df.loc[df['consistency_score'].idxmin()]
        return most_consistent.to_dict()
    
    def _get_best_risk_adjusted(self, df: pd.DataFrame) -> Dict:
        """Get best risk-adjusted performance (Sharpe / max drawdown ratio)."""
        df = df.copy()
        df['risk_adj_score'] = df['out_sharpe'] / (df['out_max_dd'] + 1e-6)  # Avoid division by zero
        
        best_risk_adj = df.loc[df['risk_adj_score'].idxmax()]
        return best_risk_adj.to_dict()
    
    @staticmethod
    def _max_drawdown(equity):
        """Calculate maximum drawdown from equity curve."""
        if not equity or len(equity) < 2:
            return 0
        arr = np.array(equity)
        highs = np.maximum.accumulate(arr)
        dd = (arr - highs)
        min_dd = dd.min()
        return -min_dd
    
    @staticmethod
    def _sharpe_ratio(equity, risk_free=0):
        """Calculate Sharpe ratio from equity curve."""
        if not equity or len(equity) < 2:
            return 0
        returns = np.diff(equity) / np.array(equity[:-1])
        excess = returns - risk_free/252
        if returns.std() == 0:
            return 0
        return np.sqrt(252) * excess.mean() / returns.std()