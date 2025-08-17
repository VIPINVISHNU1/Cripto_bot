import pandas as pd
import numpy as np
from strategy.smc_fvg_loose_strategy import SMCFVGLooseStrategy
from strategy.enhanced_smc_fvg_strategy import EnhancedSMCFVGStrategy
from strategy.advanced_smc_fvg_strategy import AdvancedSMCFVGStrategy
from strategy.optimized_smc_fvg_strategy import OptimizedSMCFVGStrategy
from strategy.high_winrate_smc_fvg_strategy import HighWinRateSMCFVGStrategy
from strategy.final_enhanced_smc_fvg_strategy import FinalEnhancedSMCFVGStrategy
from broker.mock_broker import MockBroker
from utils.logger import get_logger
from backtest.backtester import Backtester
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_strategy(strategy_class, strategy_name, config, logger):
    """Test a strategy and return results"""
    print(f"\n=== Testing {strategy_name} ===")
    
    broker = MockBroker(config["broker"], logger)
    strategy = strategy_class(config["strategy"], broker)
    
    # Patch broker for compatibility
    orig_get_historical_klines = broker.get_historical_klines
    def get_historical_klines_with_index(symbol, timeframe, start, end):
        df = orig_get_historical_klines(symbol, timeframe, start, end)
        if df is not None and "timestamp" in df.columns:
            df = df.set_index("timestamp")
        return df
    broker.get_historical_klines = get_historical_klines_with_index
    
    backtester = Backtester(config, strategy, logger)
    
    # Get data for testing
    data = broker.get_historical_klines(
        config["strategy"]["symbol"],
        config["strategy"]["timeframe"],
        config["backtest"]["start"],
        config["backtest"]["end"]
    )
    
    if data is None or len(data) == 0:
        print(f"No data available for {strategy_name}")
        return None
    
    # Split data
    split_idx = int(len(data) * 0.7)
    data_in_sample = data.iloc[:split_idx]
    data_out_sample = data.iloc[split_idx:]
    
    # Run backtests
    results_in = backtester.simulate_trades(data_in_sample, "test_in")
    results_out = backtester.simulate_trades(data_out_sample, "test_out")
    
    # Calculate metrics
    def calc_metrics(results):
        trades = results["trades"]
        if not trades:
            return {"win_rate": 0, "total_pnl": 0, "num_trades": 0, "sharpe": 0, "max_dd": 0}
        
        pnl = [t["pnl"] for t in trades]
        wins = len([p for p in pnl if p > 0])
        win_rate = wins / len(pnl) if pnl else 0
        total_pnl = sum(pnl)
        
        eq_curve = results["equity_curve"]
        max_dd, _ = backtester.max_drawdown(eq_curve) if eq_curve else (0, 0)
        sharpe = backtester.sharpe_ratio(eq_curve) if eq_curve else 0
        
        return {
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "num_trades": len(trades),
            "sharpe": sharpe,
            "max_dd": max_dd
        }
    
    in_metrics = calc_metrics(results_in)
    out_metrics = calc_metrics(results_out)
    
    print(f"In-sample: Win Rate: {in_metrics['win_rate']:.1%}, P&L: {in_metrics['total_pnl']:.2f}, Trades: {in_metrics['num_trades']}, Sharpe: {in_metrics['sharpe']:.2f}")
    print(f"Out-sample: Win Rate: {out_metrics['win_rate']:.1%}, P&L: {out_metrics['total_pnl']:.2f}, Trades: {out_metrics['num_trades']}, Sharpe: {out_metrics['sharpe']:.2f}")
    
    return {
        "strategy": strategy_name,
        "in_sample": in_metrics,
        "out_sample": out_metrics
    }

def main():
    config = load_config()
    logger = get_logger(config["logging"])
    
    strategies = [
        (SMCFVGLooseStrategy, "Baseline SMC FVG Loose"),
        (EnhancedSMCFVGStrategy, "Enhanced SMC FVG"),
        (AdvancedSMCFVGStrategy, "Advanced SMC FVG"),
        (OptimizedSMCFVGStrategy, "Optimized SMC FVG"),
        (HighWinRateSMCFVGStrategy, "High Win Rate SMC FVG"),
        (FinalEnhancedSMCFVGStrategy, "Final Enhanced SMC FVG")
    ]
    
    results = []
    
    for strategy_class, strategy_name in strategies:
        try:
            result = test_strategy(strategy_class, strategy_name, config, logger)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error testing {strategy_name}: {e}")
    
    # Summary report
    print("\n" + "="*80)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Strategy':<25} {'In-Sample':<20} {'Out-Sample':<20} {'Combined':<15}")
    print(f"{'Name':<25} {'WR | P&L | Trades':<20} {'WR | P&L | Trades':<20} {'Avg WR':<15}")
    print("-" * 80)
    
    for result in results:
        in_sample = result["in_sample"]
        out_sample = result["out_sample"]
        
        in_wr = in_sample["win_rate"]
        in_pnl = in_sample["total_pnl"]
        in_trades = in_sample["num_trades"]
        
        out_wr = out_sample["win_rate"]
        out_pnl = out_sample["total_pnl"]
        out_trades = out_sample["num_trades"]
        
        # Calculate combined win rate (weighted by number of trades)
        total_trades = in_trades + out_trades
        if total_trades > 0:
            combined_wr = (in_wr * in_trades + out_wr * out_trades) / total_trades
        else:
            combined_wr = 0
        
        print(f"{result['strategy']:<25} {in_wr:.1%}|{in_pnl:+6.2f}|{in_trades:2d}     {out_wr:.1%}|{out_pnl:+6.2f}|{out_trades:2d}     {combined_wr:.1%}")
    
    # Find best strategy for 80% target
    print("\n" + "="*80)
    print("ANALYSIS FOR 80% WIN RATE TARGET")
    print("="*80)
    
    best_strategies = []
    for result in results:
        in_sample = result["in_sample"]
        out_sample = result["out_sample"]
        
        # Calculate overall performance score
        total_trades = in_sample["num_trades"] + out_sample["num_trades"]
        if total_trades >= 3:  # Minimum trades for significance
            combined_wr = (in_sample["win_rate"] * in_sample["num_trades"] + 
                          out_sample["win_rate"] * out_sample["num_trades"]) / total_trades
            combined_pnl = in_sample["total_pnl"] + out_sample["total_pnl"]
            
            best_strategies.append({
                "name": result["strategy"],
                "win_rate": combined_wr,
                "total_pnl": combined_pnl,
                "total_trades": total_trades,
                "score": combined_wr * 0.7 + (combined_pnl > 0) * 0.3  # Weighted score
            })
    
    if best_strategies:
        best_strategies.sort(key=lambda x: x["score"], reverse=True)
        
        print("Ranking by performance score (win rate * 0.7 + profitability * 0.3):")
        for i, strategy in enumerate(best_strategies, 1):
            status = "✓ ACHIEVES TARGET" if strategy["win_rate"] >= 0.8 else "Below target"
            print(f"{i}. {strategy['name']:<30} Win Rate: {strategy['win_rate']:.1%} ({status})")
            print(f"   Total P&L: {strategy['total_pnl']:+.2f}, Trades: {strategy['total_trades']}, Score: {strategy['score']:.3f}")
    
    print(f"\nTarget: 80% win rate - {'ACHIEVED' if any(s['win_rate'] >= 0.8 for s in best_strategies) else 'NOT YET ACHIEVED'}")

if __name__ == "__main__":
    main()